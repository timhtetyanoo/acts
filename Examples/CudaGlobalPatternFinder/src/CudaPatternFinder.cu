// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFinding/CudaPatternFinder.hpp"
#include "ActsExamples/Utilities/CudaUtilities.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <limits>
#include <stdexcept>

#include "CudaPatternBuilder.cuh"

namespace {

using namespace ActsExamples;
namespace Math = CudaPatternMath;

/// The candidate checks of GlobalPatternFinder::findPatternsInEta for one
/// seed, in the same order. Returns the CudaSeedStatus
__device__ std::uint8_t prepareSeed(const CudaHitPayloadArrays& hits,
                                    const CudaCandidateArrays& lists,
                                    const CudaPatternFinderConfig& cfg,
                                    std::uint32_t seed) {
  std::uint32_t* rows{lists.candidateIndices + lists.begin(seed)};
  std::uint16_t* globLayer{lists.candidateGlobLayer + lists.begin(seed)};
  const std::uint32_t n{lists.end(seed) - lists.begin(seed)};
  const std::uint32_t minLayers{cfg.minStripEtaLayers + cfg.minPrecisionLayers};
  if (n == 0u || n < minLayers) {
    return CudaSeedStatus::tooFewHits;
  }

  // The counters have the width of PatternTopology::GroupIdx, as on the CPU
  std::uint8_t groupCounts[Math::Station::nGroups]{};
  std::uint8_t nValidGroups{0u};
  for (std::uint32_t i = 0u; i < n; ++i) {
    const std::uint8_t groupIdx{Math::groupIndex(Math::loadHit(hits, rows[i]))};
    if (++groupCounts[groupIdx] == cfg.minGroupLayers) {
      ++nValidGroups;
    }
    if (nValidGroups >= cfg.minGroups) {
      break;
    }
  }
  if (nValidGroups < cfg.minGroups) {
    return CudaSeedStatus::tooFewGroups;
  }
  if (n >= cudaInvalidHitPosition) {
    return CudaSeedStatus::overflow;
  }

  // Stable insertion sort by layer
  for (std::uint32_t i = 1u; i < n; ++i) {
    const std::uint32_t row{rows[i]};
    const Math::Hit hit{Math::loadHit(hits, row)};
    std::uint32_t j{i};
    while (j > 0u &&
           Math::layerSorter(hit, Math::loadHit(hits, rows[j - 1u]))) {
      rows[j] = rows[j - 1u];
      --j;
    }
    rows[j] = row;
  }

  globLayer[0] = 0u;
  for (std::uint32_t i = 1u; i < n; ++i) {
    globLayer[i] =
        globLayer[i - 1u] + !Math::sameLayer(Math::loadHit(hits, rows[i - 1u]),
                                             Math::loadHit(hits, rows[i]));
  }
  if (globLayer[n - 1u] + 1u < minLayers) {
    return CudaSeedStatus::tooFewLayers;
  }

  lists.seedPosition[seed] = cudaInvalidHitPosition;
  for (std::uint32_t i = 0u; i < n; ++i) {
    if (rows[i] == lists.seedHitIndex[seed]) {
      lists.seedPosition[seed] = static_cast<std::uint16_t>(i);
      break;
    }
  }
  return CudaSeedStatus::accepted;
}

__global__ void prepareSeedsKernel(CudaHitPayloadArrays hits,
                                   CudaCandidateArrays lists,
                                   CudaPatternFinderConfig cfg) {
  const std::uint32_t seed{blockIdx.x * blockDim.x + threadIdx.x};
  if (seed >= lists.nSeeds) {
    return;
  }
  lists.seedStatus[seed] = prepareSeed(hits, lists, cfg, seed);
}

/// The working memory of one batch of seeds: entry b of the batch is seed
/// firstSeed + b and owns SeedBuilder::nRegions * capacity patterns, each with
/// maxHits hit positions. After the build, the first nResults[b] patterns of
/// its result region are its finalized patterns, and overflow[b] holds the
/// CudaPatternOverflow bits of what did not fit; such a seed has no patterns.
struct Workspace {
  Pattern* patterns{nullptr};
  std::uint16_t* patternHits{nullptr};
  std::uint32_t* nResults{nullptr};
  std::uint8_t* overflow{nullptr};
  std::uint32_t nEntries{0u};
  std::uint32_t capacity{0u};
  std::uint32_t maxHits{0u};

  /// The bytes of one pattern slot with its hits
  __host__ __device__ static std::size_t slotBytes(std::size_t maxHits) {
    return sizeof(Pattern) + maxHits * sizeof(std::uint16_t);
  }

  /// The builder of entry b
  __device__ SeedBuilder builder(const CudaHitPayloadArrays& hits,
                                 const CudaCandidateArrays& lists,
                                 const CudaPatternFinderConfig& cfg,
                                 std::uint32_t seed, std::uint32_t b) const {
    const std::size_t nWork{SeedBuilder::nRegions *
                            static_cast<std::size_t>(capacity)};
    return SeedBuilder{
        .hits = hits,
        .cfg = cfg,
        .rows = lists.candidateIndices + lists.begin(seed),
        .globLayers = lists.candidateGlobLayer + lists.begin(seed),
        .patterns = patterns + b * nWork,
        .patternHits = patternHits + b * nWork * maxHits,
        .capacity = capacity,
        .maxHits = maxHits};
  }
};

/// Builds the patterns of the seeds of a batch, one thread per seed, into the
/// workspace
__global__ void buildPatternsKernel(CudaHitPayloadArrays hits,
                                    CudaCandidateArrays lists,
                                    CudaPatternFinderConfig cfg,
                                    std::uint32_t firstSeed, Workspace ws) {
  const std::uint32_t b{blockIdx.x * blockDim.x + threadIdx.x};
  const std::uint32_t seed{firstSeed + b};
  if (b >= ws.nEntries || seed >= lists.nSeeds) {
    return;
  }
  ws.nResults[b] = 0u;
  ws.overflow[b] = 0u;
  const std::uint16_t seedPos{lists.seedPosition[seed]};
  if (lists.seedStatus[seed] != CudaSeedStatus::accepted ||
      seedPos == cudaInvalidHitPosition) {
    return;
  }
  SeedBuilder builder{ws.builder(hits, lists, cfg, seed, b)};
  const std::uint32_t nResult{
      builder.build(seedPos, lists.end(seed) - lists.begin(seed),
                    static_cast<std::int8_t>(lists.seedSector[seed]))};
  if (builder.overflow != 0u) {
    ws.overflow[b] = builder.overflow;
    return;
  }
  // The quality cuts were applied during the build
  ws.nResults[b] = nResult;
}

/// Writes the finalized patterns of every seed of a batch to the records
__global__ void exportPatternsKernel(CudaHitPayloadArrays hits,
                                     CudaCandidateArrays lists,
                                     CudaPatternFinderConfig cfg,
                                     std::uint32_t firstSeed, Workspace ws,
                                     CudaPatternRecordArrays out) {
  const std::uint32_t b{blockIdx.x * blockDim.x + threadIdx.x};
  const std::uint32_t seed{firstSeed + b};
  if (b >= ws.nEntries || b >= out.nSeeds || seed >= lists.nSeeds) {
    return;
  }
  out.overflow[b] = ws.overflow[b];
  out.nPatterns[b] = ws.nResults[b];
  if (ws.nResults[b] == 0u) {
    return;
  }
  SeedBuilder builder{ws.builder(hits, lists, cfg, seed, b)};
  for (std::uint32_t k = 0u; k < ws.nResults[b]; ++k) {
    const Pattern& pat{builder.pattern(SeedBuilder::regionResult, k)};
    writePattern(builder, pat, builder.hitsOf(pat), out, out.slot(b, k));
  }
}

/// Takes the patterns of one batch of seeds into the event's patterns, one
/// seed after the other in tree order, as GlobalPatternFinder::insertIfBest
/// does into the output patterns. A seed is skipped when maxSeedAttempts
/// accepted patterns near it in theta and sector contain its hit. A new
/// pattern that overlaps a better accepted pattern is dropped; otherwise it is
/// accepted and the accepted patterns it overlaps are dropped. Runs in one
/// thread.
///
/// The patterns of seed b of the batch are the results of entry b of the
/// workspace. The CudaPatternOverflow bits of a seed that did not fit are
/// added to those of the event. When the event's slots are full, the dropped
/// patterns are removed to make room; if they are all accepted, the new
/// pattern is not taken and the event gets the eventPatterns bit.
__global__ void acceptPatternsKernel(CudaHitPayloadArrays hits,
                                     CudaCandidateArrays lists,
                                     CudaPatternFinderConfig cfg,
                                     std::uint32_t firstSeed, Workspace ws,
                                     CudaPatternRecordArrays event) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) {
    return;
  }
  auto eventRows = [&event](std::uint32_t slot) {
    return [&event, slot](std::uint32_t i) {
      return event.hitIndex[event.hitSlot(slot, i)];
    };
  };
  auto isAlive = [&](std::uint32_t slot) {
    return event.status[slot] == CudaPatternStatus::ok;
  };
  std::uint32_t nEvent{event.nPatterns[0]};
  for (std::uint32_t b = 0u; b < ws.nEntries; ++b) {
    const std::uint32_t seed{firstSeed + b};
    if (seed >= lists.nSeeds ||
        lists.seedStatus[seed] != CudaSeedStatus::accepted) {
      continue;
    }
    if (ws.overflow[b] != 0u) {
      event.overflow[0] |= ws.overflow[b];
      continue;
    }
    if (ws.nResults[b] == 0u) {
      continue;
    }
    // The skip rule, against the accepted patterns
    const std::uint32_t seedRow{lists.seedHitIndex[seed]};
    const auto seedSector = static_cast<std::int8_t>(lists.seedSector[seed]);
    std::uint32_t nContaining{0u};
    for (std::uint32_t e = 0u; e < nEvent; ++e) {
      if (!isAlive(e) ||
          std::abs(event.patTheta[e] - lists.seedTheta[seed]) >
              cfg.maxThetaSizeOverlap ||
          !Math::ExpandedSector::isNeighbour(event.sector[e], seedSector)) {
        continue;
      }
      for (std::uint32_t i = 0u; i < event.nHits[e]; ++i) {
        if (event.hitIndex[event.hitSlot(e, i)] == seedRow) {
          ++nContaining;
          break;
        }
      }
    }
    if (nContaining >= cfg.maxSeedAttempts) {
      continue;
    }

    SeedBuilder builder{ws.builder(hits, lists, cfg, seed, b)};
    for (std::uint32_t k = 0u; k < ws.nResults[b]; ++k) {
      const Pattern& pat{builder.pattern(SeedBuilder::regionResult, k)};
      const PatternInfo info{builder.info(pat)};
      const std::uint32_t good{builder.countGroups(pat, true)};
      // areOverlapping(other, candidate), as insertIfBest calls it
      auto overlapsWith = [&](std::uint32_t e, std::uint32_t& goodE) {
        goodE =
            countGroups(hits, event.nHits[e], eventRows(e), cfg.minGroupLayers);
        return areOverlapping(hits, cfg, recordInfo(event, e), eventRows(e),
                              goodE, info, builder.rowsOf(pat), good);
      };

      bool beaten{false};
      for (std::uint32_t e = 0u; e < nEvent && !beaten; ++e) {
        std::uint32_t goodE{0u};
        beaten = isAlive(e) && overlapsWith(e, goodE) &&
                 isBetterOverlap(recordInfo(event, e), goodE, info, good);
      }
      if (beaten) {
        continue;
      }
      if (nEvent == event.capacityPerSeed) {
        // Make room by removing the dropped patterns, keeping the order
        std::uint32_t nAlive{0u};
        for (std::uint32_t e = 0u; e < nEvent; ++e) {
          if (isAlive(e)) {
            if (e != nAlive) {
              copyRecord(event, e, event, nAlive);
            }
            ++nAlive;
          }
        }
        nEvent = nAlive;
      }
      if (nEvent == event.capacityPerSeed) {
        event.overflow[0] |= CudaPatternOverflow::eventPatterns;
        continue;
      }
      for (std::uint32_t e = 0u; e < nEvent; ++e) {
        std::uint32_t goodE{0u};
        if (isAlive(e) && overlapsWith(e, goodE)) {
          event.status[e] = CudaPatternStatus::rejected;
          event.flags[e] |= CudaPatternFlags::isOverlap;
        }
      }
      writePattern(builder, pat, builder.hitsOf(pat), event, nEvent++);
    }
  }
  event.nPatterns[0] = nEvent;
}

/// Allocates the workspace of a batch of nEntries seeds on the stream, with
/// the given room per seed
Workspace allocateWorkspace(std::size_t nEntries, std::size_t capacity,
                            std::size_t maxHits, cudaStream_t stream) {
  Workspace ws{.nEntries = static_cast<std::uint32_t>(nEntries),
               .capacity = static_cast<std::uint32_t>(capacity),
               .maxHits = static_cast<std::uint32_t>(maxHits)};
  const std::size_t nWork{nEntries * SeedBuilder::nRegions * capacity};
  ACTS_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ws.patterns),
                                  nWork * sizeof(Pattern), stream));
  ACTS_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ws.patternHits),
                                  nWork * maxHits * sizeof(std::uint16_t),
                                  stream));
  ACTS_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ws.nResults),
                                  nEntries * sizeof(std::uint32_t), stream));
  ACTS_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ws.overflow),
                                  nEntries * sizeof(std::uint8_t), stream));
  return ws;
}

void freeWorkspace(const Workspace& ws, cudaStream_t stream) {
  ACTS_CUDA_CHECK(cudaFreeAsync(ws.patterns, stream));
  ACTS_CUDA_CHECK(cudaFreeAsync(ws.patternHits, stream));
  ACTS_CUDA_CHECK(cudaFreeAsync(ws.nResults, stream));
  ACTS_CUDA_CHECK(cudaFreeAsync(ws.overflow, stream));
}

/// Launches the build of a batch
void launchBuild(const CudaHitPayloadContainer& hits,
                 const CudaCandidateListContainer& lists,
                 const CudaPatternFinderConfig& cfg, std::size_t firstSeed,
                 const Workspace& ws, cudaStream_t stream) {
  constexpr unsigned blockSize{64u};
  const auto nBlocks{
      static_cast<unsigned>((ws.nEntries + blockSize - 1u) / blockSize)};
  buildPatternsKernel<<<nBlocks, blockSize, 0, stream>>>(
      hits.deviceArrays(), lists.deviceArrays(), cfg,
      static_cast<std::uint32_t>(firstSeed), ws);
  ACTS_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

namespace ActsExamples {

void prepareCudaSeeds(const CudaHitPayloadContainer& hits,
                      CudaCandidateListContainer& lists,
                      const CudaPatternFinderConfig& cfg, cudaStream_t stream) {
  if (!hits.isOnDevice() || !lists.isOnDevice()) {
    throw std::invalid_argument(
        "prepareCudaSeeds: the hits and the candidate lists must be on the "
        "device");
  }
  if (lists.empty()) {
    return;
  }
  constexpr unsigned blockSize{128u};
  const auto nBlocks{
      static_cast<unsigned>((lists.nSeeds() + blockSize - 1u) / blockSize)};
  prepareSeedsKernel<<<nBlocks, blockSize, 0, stream>>>(
      hits.deviceArrays(), lists.deviceArrays(), cfg);
  ACTS_CUDA_CHECK(cudaGetLastError());
}

void buildCudaPatterns(const CudaHitPayloadContainer& hits,
                       const CudaCandidateListContainer& lists,
                       const CudaPatternFinderConfig& cfg,
                       std::size_t firstSeed,
                       CudaPatternRecordContainer& patterns,
                       cudaStream_t stream) {
  if (!hits.isOnDevice() || !lists.isOnDevice() || !patterns.isOnDevice()) {
    throw std::invalid_argument(
        "buildCudaPatterns: the hits, the candidate lists and the pattern "
        "records must be on the device");
  }
  if (patterns.empty() || firstSeed >= lists.nSeeds()) {
    return;
  }
  const Workspace ws{allocateWorkspace(patterns.nSeeds(),
                                       patterns.capacityPerSeed(),
                                       patterns.maxHitsPerPattern(), stream)};
  launchBuild(hits, lists, cfg, firstSeed, ws, stream);
  constexpr unsigned blockSize{64u};
  const auto nBlocks{
      static_cast<unsigned>((ws.nEntries + blockSize - 1u) / blockSize)};
  exportPatternsKernel<<<nBlocks, blockSize, 0, stream>>>(
      hits.deviceArrays(), lists.deviceArrays(), cfg,
      static_cast<std::uint32_t>(firstSeed), ws, patterns.deviceArrays());
  ACTS_CUDA_CHECK(cudaGetLastError());
  freeWorkspace(ws, stream);
}

CudaPatternRecordContainer findCudaPatternsInEta(
    const CudaHitPayloadContainer& hits, CudaCandidateListContainer& lists,
    const CudaPatternFinderConfig& cfg, const CudaPatternFinderSizes& sizes,
    cudaStream_t stream, CudaPatternFinderTiming* timing) {
  if (sizes.batchSize == 0u || sizes.maxHitsPerPattern == 0u ||
      sizes.maxHitsPerPattern >= cudaInvalidHitPosition) {
    throw std::invalid_argument(
        "findCudaPatternsInEta: the batch size and the hits per pattern have "
        "to be positive, and the hits have to fit the hit positions");
  }
  // The workspace is taken from the device's memory pool every event. The
  // pool keeps what is freed, instead of returning it to the system at each
  // synchronization, so that taking it again costs nothing
  {
    int device{0};
    ACTS_CUDA_CHECK(cudaGetDevice(&device));
    cudaMemPool_t pool{};
    ACTS_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, device));
    std::uint64_t keepAll{std::numeric_limits<std::uint64_t>::max()};
    ACTS_CUDA_CHECK(cudaMemPoolSetAttribute(
        pool, cudaMemPoolAttrReleaseThreshold, &keepAll));
  }
  // The room per seed is what the workspace budget leaves to each seed of a
  // batch, at least one pattern per region
  const std::size_t batchSize{
      std::max<std::size_t>(1u, std::min(sizes.batchSize, lists.nSeeds()))};
  const std::size_t capacity{std::max<std::size_t>(
      1u,
      sizes.workspaceBytes / (batchSize * SeedBuilder::nRegions *
                              Workspace::slotBytes(sizes.maxHitsPerPattern)))};

  // Marks on the stream: the start, the start of the download and its end
  std::array<cudaEvent_t, 3> marks{};
  if (timing != nullptr) {
    for (cudaEvent_t& mark : marks) {
      ACTS_CUDA_CHECK(cudaEventCreate(&mark));
    }
  }
  auto mark = [&](std::size_t i) {
    if (timing != nullptr) {
      ACTS_CUDA_CHECK(cudaEventRecord(marks[i], stream));
    }
  };

  mark(0u);
  prepareCudaSeeds(hits, lists, cfg, stream);
  CudaPatternRecordContainer event{1u, sizes.maxPatterns,
                                   sizes.maxHitsPerPattern};
  event.moveToDevice(stream);
  const Workspace ws{
      allocateWorkspace(batchSize, capacity, sizes.maxHitsPerPattern, stream)};
  for (std::size_t firstSeed = 0u; firstSeed < lists.nSeeds();
       firstSeed += batchSize) {
    launchBuild(hits, lists, cfg, firstSeed, ws, stream);
    acceptPatternsKernel<<<1, 1, 0, stream>>>(
        hits.deviceArrays(), lists.deviceArrays(), cfg,
        static_cast<std::uint32_t>(firstSeed), ws, event.deviceArrays());
    ACTS_CUDA_CHECK(cudaGetLastError());
  }
  freeWorkspace(ws, stream);
  mark(1u);
  event.moveToHost(stream);
  mark(2u);
  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));

  if (timing != nullptr) {
    auto elapsedUs = [&](std::size_t from, std::size_t to) {
      float ms{0.f};
      ACTS_CUDA_CHECK(cudaEventElapsedTime(&ms, marks[from], marks[to]));
      return 1000. * ms;
    };
    timing->kernelsUs = elapsedUs(0u, 1u);
    timing->downloadUs = elapsedUs(1u, 2u);
    timing->capacityPerSeed = capacity;
    timing->overflow = event.overflow(0u);
    timing->nPatternsInEta = 0u;
    for (std::uint32_t k = 0u; k < event.nPatterns(0u); ++k) {
      timing->nPatternsInEta +=
          event.pattern(event.slot(0u, k)).status == CudaPatternStatus::ok;
    }
    for (cudaEvent_t& m : marks) {
      ACTS_CUDA_CHECK(cudaEventDestroy(m));
    }
  }
  return event;
}

}  // namespace ActsExamples
