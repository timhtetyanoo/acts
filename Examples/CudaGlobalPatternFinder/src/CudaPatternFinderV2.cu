// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

/// The pattern finding in eta, version 2: a warp builds a seed one layer at a
/// time, and the patterns of the event compete by priority, independently of
/// the seed order. The tests, the residuals, the fits and the cuts are those
/// of version 1 (CudaPatternBuilder.cuh); what changes is the order in which
/// they run and the way patterns compete.

#include "ActsExamples/TrackFinding/CudaPatternFinder.hpp"
#include "ActsExamples/Utilities/CudaUtilities.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

#include "CudaPatternBuilder.cuh"

namespace {

constexpr unsigned fullMask = 0xFFFFFFFFu;
constexpr unsigned lanesPerWarp = 32u;
/// Warps of a block of the kernels that work a warp per seed
constexpr unsigned warpsPerBlock = 4u;

/// Whether pattern a comes before pattern b in the priority order: more good
/// groups, more bending layers, more precision layers, a smaller mean squared
/// normalised residual, and the smaller index. A strict total order, unlike
/// isBetter, so that the rounds below have a single answer.
__device__ bool outranks(const PatternInfo& a, std::uint32_t goodA,
                         std::uint32_t indexA, const PatternInfo& b,
                         std::uint32_t goodB, std::uint32_t indexB) {
  if (goodA != goodB) {
    return goodA > goodB;
  }
  if (a.nBendingLayers() != b.nBendingLayers()) {
    return a.nBendingLayers() > b.nBendingLayers();
  }
  if (a.nPrecisionLayers != b.nPrecisionLayers) {
    return a.nPrecisionLayers > b.nPrecisionLayers;
  }
  if (a.meanResidual2 < b.meanResidual2) {
    return true;
  }
  if (b.meanResidual2 < a.meanResidual2) {
    return false;
  }
  return indexA < indexB;
}

/// The k-th set bit of a mask, counted from the lowest
__device__ std::uint32_t nthSetBit(std::uint32_t mask, std::uint32_t k) {
  for (std::uint32_t i = 0u; i < k; ++i) {
    mask &= mask - 1u;
  }
  return static_cast<std::uint32_t>(__ffs(static_cast<int>(mask)) - 1);
}

/// Exclusive prefix sum of count(i) over i < n by the lanes of a warp, into
/// out. Returns the total, the same in every lane
template <typename Count>
__device__ std::uint32_t warpExclusiveScan(std::uint32_t n, Count count,
                                           std::uint32_t* out,
                                           std::uint32_t lane) {
  std::uint32_t carry{0u};
  for (std::uint32_t base = 0u; base < n; base += lanesPerWarp) {
    const std::uint32_t i{base + lane};
    const std::uint32_t value{i < n ? count(i) : 0u};
    std::uint32_t sum{value};
    for (std::uint32_t d = 1u; d < lanesPerWarp; d <<= 1u) {
      const std::uint32_t other{__shfl_up_sync(fullMask, sum, d)};
      if (lane >= d) {
        sum += other;
      }
    }
    if (i < n) {
      out[i] = carry + sum - value;
    }
    carry += __shfl_sync(fullMask, sum, lanesPerWarp - 1u);
  }
  __syncwarp();
  return carry;
}

/// Bits of the per-pattern flags of a layer
constexpr std::uint8_t flagDropped = 1u << 0u;
constexpr std::uint8_t flagNormalHit = 1u << 1u;

/// Decision of a pattern in the rounds
constexpr std::uint8_t undecided = 0u;
constexpr std::uint8_t kept = 1u;
constexpr std::uint8_t dropped = 2u;

/// One seed built by one warp. Every lane holds this object; the lanes share
/// the patterns and the scratch arrays of the seed, and every loop below runs
/// the same number of times in all of them. The patterns live in three
/// regions of `capacity` slots: two that the layers swap between and one for
/// the forward patterns, which later holds the results.
struct WarpSeed {
  static constexpr std::uint32_t regionA = 0u;
  static constexpr std::uint32_t regionB = 1u;
  static constexpr std::uint32_t regionFinal = 2u;
  static constexpr std::uint32_t nRegions = 3u;

  SeedBuilder b;
  /// Per pattern of the region being extended: the hits of the current
  /// chunk it accepts, those of them with low confidence, the flags
  std::uint32_t* acceptMask;
  std::uint32_t* lowMask;
  std::uint8_t* flags;
  /// Per pattern: prefix sums, or the good groups during the rounds
  std::uint32_t* offsets;
  /// Per pattern: the decisions of the rounds
  std::uint8_t* state;
  std::uint8_t* nextState;
  /// The hits of the chunk of a layer being tested, in on-chip memory
  Math::Hit* chunkHits;
  std::uint32_t lane;

  /// Whether any lane ran out of room
  __device__ bool overflowed() {
    return __any_sync(fullMask, b.overflow != 0u);
  }

  /// Copies the patterns of a region for which keep(i) holds to the start of
  /// another region, in order. Returns their number
  template <typename Keep>
  __device__ std::uint32_t compact(std::uint32_t from, std::uint32_t n,
                                   std::uint32_t to, Keep keep) {
    const std::uint32_t nOut{warpExclusiveScan(
        n, [&](std::uint32_t i) { return keep(i) ? 1u : 0u; }, offsets, lane)};
    for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
      if (keep(i)) {
        b.copy(b.pattern(from, i), b.pattern(to, offsets[i]));
      }
    }
    __syncwarp();
    return nOut;
  }

  /// The patterns of a layer, the positions [first, last) of the candidate
  /// list, tested against the n patterns of region `from`; the continued
  /// patterns are written to region `to`. Returns their number.
  ///
  ///  1. prune: a pattern that missed too many layers in this group, compared
  ///     to the others of its family, is dropped, and of the patterns of a
  ///     family that end on the same hit only the first in priority stays;
  ///  2. test: every hit of the layer against every pattern, against the line
  ///     the pattern had before the layer, 32 hits at a time;
  ///  3. rebuild: each compatible hit gives a child, the pattern with the hit.
  ///     A pattern carries on as it is unless it took a hit of normal
  ///     confidence.
  __device__ std::uint32_t extendLayer(std::uint32_t from, std::uint32_t n,
                                       std::uint32_t to, std::uint32_t first,
                                       std::uint32_t last) {
    const auto layerPos = static_cast<std::uint16_t>(first);
    const int layer{b.globLayer(layerPos)};
    const std::uint8_t layerGroup{b.group(layerPos)};
    auto missedLayers = [&](const Pattern& pat) {
      const int d{static_cast<int>(b.globLayer(pat.lastInsertedHit)) - layer};
      return static_cast<std::uint32_t>(d < 0 ? -d : d);
    };

    // 1. Prune
    for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
      const Pattern& pat{b.pattern(from, i)};
      std::uint32_t minMissed{UINT_MAX};
      std::uint32_t nFamily{0u};
      bool anyLong{false};
      for (std::uint32_t j = 0u; j < n; ++j) {
        const Pattern& other{b.pattern(from, j)};
        if (other.family != pat.family) {
          continue;
        }
        ++nFamily;
        const std::uint32_t missed{missedLayers(other)};
        minMissed = missed < minMissed ? missed : minMissed;
        anyLong = anyLong || other.nBendingLayers() > 2u;
      }
      const std::uint32_t maxMissed{b.cfg.maxMissLayersInGroup > minMissed
                                        ? b.cfg.maxMissLayersInGroup
                                        : minMissed};
      bool drop{b.group(pat.lastInsertedHit) == layerGroup &&
                missedLayers(pat) > maxMissed};
      if (!drop && nFamily > 1u && anyLong) {
        const PatternInfo info{b.info(pat)};
        for (std::uint32_t j = 0u; j < n && !drop; ++j) {
          const Pattern& other{b.pattern(from, j)};
          drop = j != i && other.family == pat.family &&
                 other.lastInsertedHit == pat.lastInsertedHit &&
                 outranks(b.info(other), 0u, j, info, 0u, i);
        }
      }
      flags[i] = drop ? flagDropped : 0u;
    }
    __syncwarp();
    // The line of each pattern for this layer, as the first test on a new
    // layer updates it on the CPU
    for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
      if ((flags[i] & flagDropped) == 0u) {
        b.updateLineParameters(b.pattern(from, i));
      }
    }
    __syncwarp();

    // 2. and 3., 32 hits at a time
    std::uint32_t nOut{0u};
    for (std::uint32_t chunk = first; chunk < last; chunk += lanesPerWarp) {
      const std::uint32_t m{last - chunk < lanesPerWarp ? last - chunk
                                                        : lanesPerWarp};
      for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
        acceptMask[i] = 0u;
        lowMask[i] = 0u;
      }
      if (lane < m) {
        chunkHits[lane] = b.hit(static_cast<std::uint16_t>(chunk + lane));
      }
      __syncwarp();
      for (std::uint32_t k = lane; k < n * m; k += lanesPerWarp) {
        const std::uint32_t i{k / m};
        const std::uint32_t h{k % m};
        if ((flags[i] & flagDropped) != 0u) {
          continue;
        }
        const LineTest res{b.checkLine(b.pattern(from, i),
                                       static_cast<std::uint16_t>(chunk + h),
                                       chunkHits[h])};
        if (res.decision != Decision::addHit) {
          continue;
        }
        atomicOr(&acceptMask[i], 1u << h);
        if (res.sigma > b.cfg.lowConfidenceResSigma &&
            res.residual / res.sigma > 2.) {
          atomicOr(&lowMask[i], 1u << h);
        }
      }
      __syncwarp();
      const std::uint32_t nChildren{warpExclusiveScan(
          n, [&](std::uint32_t i) { return __popc(acceptMask[i]); }, offsets,
          lane)};
      if (nOut + nChildren > b.capacity) {
        b.overflow |= CudaPatternOverflow::patterns;
        return 0u;
      }
      for (std::uint32_t t = lane; t < nChildren; t += lanesPerWarp) {
        // The parent is the last pattern whose offset is not above t
        std::uint32_t lo{0u};
        std::uint32_t hi{n};
        while (hi - lo > 1u) {
          const std::uint32_t mid{(lo + hi) / 2u};
          if (offsets[mid] <= t) {
            lo = mid;
          } else {
            hi = mid;
          }
        }
        const std::uint32_t h{nthSetBit(acceptMask[lo], t - offsets[lo])};
        const auto test = static_cast<std::uint16_t>(chunk + h);
        Pattern& child{b.pattern(to, nOut + t)};
        b.copy(b.pattern(from, lo), child);
        b.addHit(child, test, b.checkLine(child, test, chunkHits[h]));
      }
      for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
        if ((acceptMask[i] & ~lowMask[i]) != 0u) {
          flags[i] |= flagNormalHit;
        }
      }
      nOut += nChildren;
      __syncwarp();
      if (overflowed()) {
        return 0u;
      }
    }

    // The patterns that carry on without a hit of this layer
    auto carries = [&](std::uint32_t i) {
      return (flags[i] & (flagDropped | flagNormalHit)) == 0u;
    };
    const std::uint32_t nCarried{warpExclusiveScan(
        n, [&](std::uint32_t i) { return carries(i) ? 1u : 0u; }, offsets,
        lane)};
    if (nOut + nCarried > b.capacity) {
      b.overflow |= CudaPatternOverflow::patterns;
      return 0u;
    }
    for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
      if (carries(i)) {
        b.copy(b.pattern(from, i), b.pattern(to, nOut + offsets[i]));
      }
    }
    __syncwarp();
    return nOut + nCarried;
  }

  /// The overlaps among the n patterns of a region, by priority: in each
  /// round, a pattern that no undecided or kept pattern above it overlaps is
  /// kept, and one that a kept pattern above it overlaps is dropped. With
  /// perFamily, only patterns of the same family compete. The survivors are
  /// copied to region `to`; returns their number
  __device__ std::uint32_t resolveOverlaps(std::uint32_t from, std::uint32_t n,
                                           std::uint32_t to, bool perFamily) {
    std::uint32_t* good{offsets};
    for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
      state[i] = undecided;
      good[i] = b.countGroups(b.pattern(from, i), true);
    }
    __syncwarp();
    while (true) {
      bool anyUndecided{false};
      for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
        nextState[i] = state[i];
        if (state[i] != undecided) {
          continue;
        }
        const Pattern& pat{b.pattern(from, i)};
        const PatternInfo info{b.info(pat)};
        bool isDropped{false};
        bool isBlocked{false};
        for (std::uint32_t j = 0u; j < n && !isDropped; ++j) {
          const Pattern& other{b.pattern(from, j)};
          if (j == i || state[j] == dropped ||
              (perFamily && other.family != pat.family) ||
              !outranks(b.info(other), good[j], j, info, good[i], i) ||
              !b.areOverlapping(other, pat)) {
            continue;
          }
          isDropped = state[j] == kept;
          isBlocked = true;
        }
        nextState[i] = isDropped ? dropped : (isBlocked ? undecided : kept);
        anyUndecided = anyUndecided || nextState[i] == undecided;
      }
      __syncwarp();
      for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
        state[i] = nextState[i];
      }
      __syncwarp();
      if (!__any_sync(fullMask, anyUndecided)) {
        break;
      }
    }
    return compact(from, n, to,
                   [&](std::uint32_t i) { return state[i] == kept; });
  }

  /// Builds the patterns of a seed into the final region. Returns their
  /// number, zero if the seed did not fit
  __device__ std::uint32_t build(std::uint16_t seedPos, std::uint32_t nCands,
                                 std::int8_t sector) {
    if (lane == 0u) {
      b.makeSeedPattern(b.pattern(regionA, 0u), seedPos, sector);
    }
    __syncwarp();
    const std::uint16_t seedLayer{b.globLayer(seedPos)};
    std::uint32_t current{regionA};
    std::uint32_t next{regionB};
    std::uint32_t n{1u};

    // Forward: the layers after the seed layer, outwards
    std::uint32_t first{seedPos + 1u};
    while (first < nCands && b.globLayer(first) == seedLayer) {
      ++first;
    }
    while (first < nCands) {
      std::uint32_t last{first + 1u};
      while (last < nCands && b.globLayer(last) == b.globLayer(first)) {
        ++last;
      }
      n = extendLayer(current, n, next, first, last);
      if (overflowed()) {
        return 0u;
      }
      const std::uint32_t swap{current};
      current = next;
      next = swap;
      first = last;
    }
    std::uint32_t nForward{1u};
    if (n > 1u) {
      nForward = resolveOverlaps(current, n, regionFinal, false);
    } else if (lane == 0u) {
      b.copy(b.pattern(current, 0u), b.pattern(regionFinal, 0u));
    }
    __syncwarp();

    // Backward: every forward pattern from the seed inwards, side by side.
    // Each is a family of its own
    for (std::uint32_t f = lane; f < nForward; f += lanesPerWarp) {
      Pattern& start{b.pattern(regionA, f)};
      b.copy(b.pattern(regionFinal, f), start);
      b.moveLineAnchorHit(start, seedPos);
      start.lastInsertedHit = seedPos;
      start.family = static_cast<std::uint16_t>(f);
    }
    __syncwarp();
    current = regionA;
    next = regionB;
    n = nForward;
    std::uint32_t last{seedPos};
    while (last > 0u && b.globLayer(last - 1u) == seedLayer) {
      --last;
    }
    while (last > 0u) {
      std::uint32_t begin{last - 1u};
      while (begin > 0u && b.globLayer(begin - 1u) == b.globLayer(last - 1u)) {
        --begin;
      }
      n = extendLayer(current, n, next, begin, last);
      if (overflowed()) {
        return 0u;
      }
      const std::uint32_t swap{current};
      current = next;
      next = swap;
      last = begin;
    }

    // The overlaps inside each family, as at the end of each backward pass of
    // the CPU, then among all of them
    if (n > 1u) {
      n = resolveOverlaps(current, n, next, true);
      const std::uint32_t swap{current};
      current = next;
      next = swap;
    }
    if (n > 1u) {
      n = resolveOverlaps(current, n, next, false);
      const std::uint32_t swap{current};
      current = next;
      next = swap;
    }

    // Finalize and keep the patterns that pass the cuts
    for (std::uint32_t k = lane; k < n; k += lanesPerWarp) {
      Pattern& pat{b.pattern(current, k)};
      pat.meanNormResidual2 /= pat.nDoF;
      flags[k] = b.passPatternCuts(pat) ? 1u : 0u;
    }
    __syncwarp();
    return compact(current, n, regionFinal,
                   [&](std::uint32_t k) { return flags[k] != 0u; });
  }
};

/// The working memory of one batch of seeds: entry b of the batch is seed
/// firstSeed + b and owns WarpSeed::nRegions * capacity patterns, each with
/// maxHits hit positions, and the per-pattern scratch of `capacity` entries.
/// After the build, the first nResults[b] patterns of its final region are
/// its finalized patterns, and overflow[b] holds the CudaPatternOverflow bits
/// of what did not fit; such a seed has no patterns.
struct WorkspaceV2 {
  Pattern* patterns{nullptr};
  std::uint16_t* patternHits{nullptr};
  std::uint32_t* acceptMask{nullptr};
  std::uint32_t* lowMask{nullptr};
  std::uint32_t* offsets{nullptr};
  std::uint8_t* flags{nullptr};
  std::uint8_t* state{nullptr};
  std::uint8_t* nextState{nullptr};
  std::uint32_t* nResults{nullptr};
  std::uint8_t* overflow{nullptr};
  /// Start of each entry's patterns in the event pool
  std::uint32_t* poolOffsets{nullptr};
  std::uint32_t nEntries{0u};
  std::uint32_t capacity{0u};
  std::uint32_t maxHits{0u};

  /// The bytes of one pattern of capacity: its slots in every region, with
  /// their hits, and its scratch
  __host__ __device__ static std::size_t bytesPerPattern(std::size_t maxHits) {
    return WarpSeed::nRegions *
               (sizeof(Pattern) + maxHits * sizeof(std::uint16_t)) +
           3u * sizeof(std::uint32_t) + 3u * sizeof(std::uint8_t);
  }

  /// The builder of entry b
  __device__ SeedBuilder builder(const CudaHitPayloadArrays& hits,
                                 const CudaCandidateArrays& lists,
                                 const CudaPatternFinderConfig& cfg,
                                 std::uint32_t seed, std::uint32_t b) const {
    const std::size_t nWork{WarpSeed::nRegions *
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

  /// The warp of entry b
  __device__ WarpSeed warpSeed(const CudaHitPayloadArrays& hits,
                               const CudaCandidateArrays& lists,
                               const CudaPatternFinderConfig& cfg,
                               std::uint32_t seed, std::uint32_t b,
                               std::uint32_t lane, Math::Hit* chunkHits) const {
    const std::size_t base{b * static_cast<std::size_t>(capacity)};
    return WarpSeed{.b = builder(hits, lists, cfg, seed, b),
                    .acceptMask = acceptMask + base,
                    .lowMask = lowMask + base,
                    .flags = flags + base,
                    .offsets = offsets + base,
                    .state = state + base,
                    .nextState = nextState + base,
                    .chunkHits = chunkHits,
                    .lane = lane};
  }
};

/// Builds the seeds of a batch, one warp per seed
__global__ void buildPatternsKernelV2(CudaHitPayloadArrays hits,
                                      CudaCandidateArrays lists,
                                      CudaPatternFinderConfig cfg,
                                      std::uint32_t firstSeed, WorkspaceV2 ws) {
  const std::uint32_t b{(blockIdx.x * blockDim.x + threadIdx.x) / lanesPerWarp};
  const std::uint32_t lane{threadIdx.x % lanesPerWarp};
  const std::uint32_t seed{firstSeed + b};
  if (b >= ws.nEntries) {
    return;
  }
  if (lane == 0u) {
    ws.nResults[b] = 0u;
    ws.overflow[b] = 0u;
  }
  if (seed >= lists.nSeeds ||
      lists.seedStatus[seed] != CudaSeedStatus::accepted ||
      lists.seedPosition[seed] == cudaInvalidHitPosition) {
    return;
  }
  // The hits of the chunk being tested, per warp of the block. Raw storage,
  // since shared memory takes no constructors
  __shared__ alignas(Math::Hit) unsigned char
      chunkStorage[warpsPerBlock * lanesPerWarp * sizeof(Math::Hit)];
  Math::Hit* chunkHits{reinterpret_cast<Math::Hit*>(chunkStorage) +
                       (threadIdx.x / lanesPerWarp) * lanesPerWarp};
  WarpSeed warp{ws.warpSeed(hits, lists, cfg, seed, b, lane, chunkHits)};
  const std::uint32_t nResult{
      warp.build(lists.seedPosition[seed], lists.end(seed) - lists.begin(seed),
                 static_cast<std::int8_t>(lists.seedSector[seed]))};
  const auto overflow = static_cast<std::uint8_t>(
      __reduce_or_sync(fullMask, static_cast<unsigned>(warp.b.overflow)));
  if (lane == 0u) {
    ws.overflow[b] = overflow;
    ws.nResults[b] = overflow != 0u ? 0u : nResult;
  }
}

/// Appends the patterns of a batch to the event pool, in seed order. The
/// patterns that find no room are left out and the event gets the
/// eventPatterns bit. Runs in one block
__global__ void appendPatternsKernel(CudaHitPayloadArrays hits,
                                     CudaCandidateArrays lists,
                                     CudaPatternFinderConfig cfg,
                                     std::uint32_t firstSeed, WorkspaceV2 ws,
                                     CudaPatternRecordArrays event) {
  __shared__ std::uint32_t nBefore;
  __shared__ std::uint32_t nBatch;
  if (threadIdx.x == 0u) {
    nBefore = event.nPatterns[0];
    std::uint32_t sum{0u};
    std::uint8_t overflow{0u};
    for (std::uint32_t b = 0u; b < ws.nEntries; ++b) {
      ws.poolOffsets[b] = sum;
      sum += ws.nResults[b];
      overflow |= ws.overflow[b];
    }
    if (nBefore + sum > event.capacityPerSeed) {
      overflow |= CudaPatternOverflow::eventPatterns;
      sum = event.capacityPerSeed - nBefore;
    }
    nBatch = sum;
    event.overflow[0] |= overflow;
    event.nPatterns[0] = nBefore + sum;
  }
  __syncthreads();
  for (std::uint32_t t = threadIdx.x; t < nBatch; t += blockDim.x) {
    std::uint32_t lo{0u};
    std::uint32_t hi{ws.nEntries};
    while (hi - lo > 1u) {
      const std::uint32_t mid{(lo + hi) / 2u};
      if (ws.poolOffsets[mid] <= t) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    SeedBuilder builder{ws.builder(hits, lists, cfg, firstSeed + lo, lo)};
    const Pattern& pat{
        builder.pattern(WarpSeed::regionFinal, t - ws.poolOffsets[lo])};
    writePattern(builder, pat, builder.hitsOf(pat), event, nBefore + t);
  }
}

/// The link of one pair of the event's patterns
__device__ void linkPair(const CudaHitPayloadArrays& hits,
                         const CudaPatternFinderConfig& cfg,
                         const CudaPatternRecordArrays& event,
                         const std::uint32_t* good, std::uint32_t* links,
                         std::uint32_t wordsPerRow, std::uint32_t i,
                         std::uint32_t j) {
  auto rowsOf = [&event](std::uint32_t slot) {
    return [&event, slot](std::uint32_t k) {
      return event.hitIndex[event.hitSlot(slot, k)];
    };
  };
  const PatternInfo infoI{recordInfo(event, i)};
  const PatternInfo infoJ{recordInfo(event, j)};
  const bool iFirst{outranks(infoI, good[i], i, infoJ, good[j], j)};
  const std::uint32_t upper{iFirst ? i : j};
  const std::uint32_t lower{iFirst ? j : i};
  if (areOverlapping(hits, cfg, iFirst ? infoI : infoJ, rowsOf(upper),
                     good[upper], iFirst ? infoJ : infoI, rowsOf(lower),
                     good[lower])) {
    atomicOr(&links[lower * wordsPerRow + upper / 32u], 1u << (upper % 32u));
  }
}

/// The links between the patterns of the event: bit j of row i is set if
/// pattern j outranks pattern i and overlaps it, by the overlap test of the
/// pattern that outranks. The threads of the grid stride over the pairs; the
/// rows must be zeroed
__global__ void linkEventPatternsKernel(CudaHitPayloadArrays hits,
                                        CudaPatternFinderConfig cfg,
                                        CudaPatternRecordArrays event,
                                        const std::uint32_t* good,
                                        std::uint32_t* links,
                                        std::uint32_t wordsPerRow) {
  const std::uint32_t n{event.nPatterns[0]};
  const std::uint64_t nPairs{static_cast<std::uint64_t>(n) * n};
  for (std::uint64_t pair =
           blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
       pair < nPairs;
       pair += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
    const auto i = static_cast<std::uint32_t>(pair / n);
    const auto j = static_cast<std::uint32_t>(pair % n);
    // The cheap part of the overlap test first
    if (j <= i ||
        !Math::ExpandedSector::isNeighbour(event.sector[i], event.sector[j]) ||
        std::abs(event.patTheta[i] - event.patTheta[j]) >
            2. * cfg.maxThetaSizeOverlap) {
      continue;
    }
    linkPair(hits, cfg, event, good, links, wordsPerRow, i, j);
  }
}

/// The good groups of each pattern of the event
__global__ void countEventGroupsKernel(CudaHitPayloadArrays hits,
                                       CudaPatternFinderConfig cfg,
                                       CudaPatternRecordArrays event,
                                       std::uint32_t* good) {
  const std::uint32_t i{blockIdx.x * blockDim.x + threadIdx.x};
  if (i >= event.nPatterns[0]) {
    return;
  }
  good[i] = countGroups(
      hits, event.nHits[i],
      [&event, i](std::uint32_t k) {
        return event.hitIndex[event.hitSlot(i, k)];
      },
      cfg.minGroupLayers);
}

/// The overlaps among the patterns of the event, by priority, as
/// WarpSeed::resolveOverlaps does it for a seed, on the links. The dropped
/// patterns get the status rejected. Runs in one block
__global__ void resolveEventOverlapsKernel(CudaPatternRecordArrays event,
                                           const std::uint32_t* links,
                                           std::uint32_t wordsPerRow,
                                           std::uint8_t* state,
                                           std::uint8_t* nextState) {
  const std::uint32_t n{event.nPatterns[0]};
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    state[i] = undecided;
  }
  __syncthreads();
  while (true) {
    bool anyUndecided{false};
    for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
      nextState[i] = state[i];
      if (state[i] != undecided) {
        continue;
      }
      bool isDropped{false};
      bool isBlocked{false};
      for (std::uint32_t w = 0u; w < wordsPerRow && !isDropped; ++w) {
        std::uint32_t word{links[i * wordsPerRow + w]};
        while (word != 0u && !isDropped) {
          const std::uint32_t j{32u * w +
                                static_cast<std::uint32_t>(__ffs(word) - 1)};
          word &= word - 1u;
          isDropped = state[j] == kept;
          isBlocked = isBlocked || state[j] == undecided;
        }
      }
      nextState[i] = isDropped ? dropped : (isBlocked ? undecided : kept);
      anyUndecided = anyUndecided || nextState[i] == undecided;
    }
    __syncthreads();
    for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
      state[i] = nextState[i];
    }
    if (__syncthreads_or(anyUndecided ? 1 : 0) == 0) {
      break;
    }
  }
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    if (state[i] == dropped) {
      event.status[i] = CudaPatternStatus::rejected;
      event.flags[i] |= CudaPatternFlags::isOverlap;
    }
  }
}

/// The quantities of a hit that PatternTopology::layerSorter and sameLayer
/// read, without the rest of the payload
__device__ Math::Hit loadSortHit(const CudaHitPayloadArrays& hits,
                                 std::uint32_t row) {
  Math::Hit hit{};
  hit.position = {hits.positionX[row], hits.positionY[row],
                  hits.positionZ[row]};
  hit.localPositionY = hits.localPositionY[row];
  hit.stationKey = hits.stationKey[row];
  hit.station = hits.station[row];
  hit.locLayer = hits.locLayer[row];
  return hit;
}

/// The candidate checks of GlobalPatternFinder::findPatternsInEta for one
/// seed, by the lanes of a warp: the group count as a reduction, the sort by
/// layer as a sorting network over the list padded to a power of two, and
/// the global layer numbers as a prefix sum over the new-layer flags. The
/// outputs are those of the version 1 kernel
__global__ void prepareSeedsKernelV2(CudaHitPayloadArrays hits,
                                     CudaCandidateArrays lists,
                                     CudaPatternFinderConfig cfg) {
  const std::uint32_t seed{(blockIdx.x * blockDim.x + threadIdx.x) /
                           lanesPerWarp};
  const std::uint32_t lane{threadIdx.x % lanesPerWarp};
  if (seed >= lists.nSeeds) {
    return;
  }
  std::uint32_t* rows{lists.candidateIndices + lists.begin(seed)};
  std::uint16_t* globLayer{lists.candidateGlobLayer + lists.begin(seed)};
  const std::uint32_t n{lists.end(seed) - lists.begin(seed)};
  const std::uint32_t minLayers{cfg.minStripEtaLayers + cfg.minPrecisionLayers};
  auto setStatus = [&](std::uint8_t status) {
    if (lane == 0u) {
      lists.seedStatus[seed] = status;
    }
  };
  if (n == 0u || n < minLayers) {
    setStatus(CudaSeedStatus::tooFewHits);
    return;
  }

  // Groups with at least minGroupLayers candidates
  std::uint32_t counts[Math::Station::nGroups]{};
  for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
    ++counts[hits.station[rows[i]]];
  }
  std::uint32_t nValidGroups{0u};
  for (std::uint32_t g = 0u; g < Math::Station::nGroups; ++g) {
    nValidGroups +=
        __reduce_add_sync(fullMask, counts[g]) >= cfg.minGroupLayers;
  }
  if (nValidGroups < cfg.minGroups) {
    setStatus(CudaSeedStatus::tooFewGroups);
    return;
  }
  if (n >= cudaInvalidHitPosition) {
    setStatus(CudaSeedStatus::overflow);
    return;
  }

  // Sort by layer. In this form of the bitonic network every comparison puts
  // the smaller element first, so the padding beyond n, larger than all, is
  // never touched
  std::uint32_t padded{1u};
  while (padded < n) {
    padded <<= 1u;
  }
  for (std::uint32_t k = 2u; k <= padded; k <<= 1u) {
    for (std::uint32_t j = k / 2u; j > 0u; j >>= 1u) {
      for (std::uint32_t t = lane; t < padded; t += lanesPerWarp) {
        const std::uint32_t partner{j == k / 2u ? t ^ (k - 1u) : t ^ j};
        if (partner <= t || partner >= n) {
          continue;
        }
        const std::uint32_t rowT{rows[t]};
        const std::uint32_t rowP{rows[partner]};
        if (Math::layerSorter(loadSortHit(hits, rowP),
                              loadSortHit(hits, rowT))) {
          rows[t] = rowP;
          rows[partner] = rowT;
        }
      }
      __syncwarp();
    }
  }

  // Global layer numbers
  std::uint32_t carry{0u};
  for (std::uint32_t base = 0u; base < n; base += lanesPerWarp) {
    const std::uint32_t i{base + lane};
    const std::uint32_t newLayer{
        i < n && i > 0u &&
        !Math::sameLayer(loadSortHit(hits, rows[i - 1u]),
                         loadSortHit(hits, rows[i]))};
    std::uint32_t sum{newLayer};
    for (std::uint32_t d = 1u; d < lanesPerWarp; d <<= 1u) {
      const std::uint32_t other{__shfl_up_sync(fullMask, sum, d)};
      if (lane >= d) {
        sum += other;
      }
    }
    if (i < n) {
      globLayer[i] = static_cast<std::uint16_t>(carry + sum);
    }
    carry += __shfl_sync(fullMask, sum, lanesPerWarp - 1u);
  }
  if (carry + 1u < minLayers) {
    setStatus(CudaSeedStatus::tooFewLayers);
    return;
  }

  // The first position of the seed hit in its sorted list
  std::uint32_t seedPos{cudaInvalidHitPosition};
  for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
    if (rows[i] == lists.seedHitIndex[seed] && i < seedPos) {
      seedPos = i;
    }
  }
  seedPos = __reduce_min_sync(fullMask, seedPos);
  if (lane == 0u) {
    lists.seedPosition[seed] = static_cast<std::uint16_t>(seedPos);
  }
  setStatus(CudaSeedStatus::accepted);
}

/// The column-major rotation of a surface or a bucket times a vector
__device__ Math::Vector3 rotate(const double* rotation, double x, double y,
                                double z) {
  return {rotation[0] * x + rotation[3] * y + rotation[6] * z,
          rotation[1] * x + rotation[4] * y + rotation[7] * z,
          rotation[2] * x + rotation[5] * y + rotation[8] * z};
}
/// Column c of a column-major rotation
__device__ Math::Vector3 column(const double* rotation, std::uint32_t c) {
  return {rotation[3u * c], rotation[3u * c + 1u], rotation[3u * c + 2u]};
}
__device__ Math::Vector3 cross(const Math::Vector3& a, const Math::Vector3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

/// The raw MuonSpacePoint::MuonId fields the pattern finding reads, with the
/// bit layout of MuonSpacePoint.cpp
namespace MuonIdBits {
__device__ std::uint32_t stationName(std::uint32_t raw) {
  return raw & 0xFu;
}
__device__ bool sideA(std::uint32_t raw) {
  return ((raw >> 4u) & 1u) != 0u;
}
__device__ std::uint32_t technology(std::uint32_t raw) {
  return (raw >> 5u) & 0x7u;
}
__device__ std::uint32_t sector(std::uint32_t raw) {
  return ((raw >> 8u) & 0x3Fu) + 1u;
}
__device__ bool measuresEta(std::uint32_t raw) {
  return ((raw >> 14u) & 1u) != 0u;
}
__device__ bool measuresPhi(std::uint32_t raw) {
  return ((raw >> 15u) & 1u) != 0u;
}
__device__ std::uint32_t detLayer(std::uint32_t raw) {
  return ((raw >> 17u) & 0xFu) + 1u;
}
/// MuonId::TechField
constexpr std::uint32_t mdt = 0u;
constexpr std::uint32_t tgc = 3u;
constexpr std::uint32_t sTgc = 4u;
constexpr std::uint32_t mm = 5u;
/// toStationIndex of the StationName order BIS, BIL, BMS, BML, BOS, BOL, BEE,
/// EIS, EIL, EMS, EML, EOS, EOL, EES, EEL
__device__ std::uint8_t station(std::uint32_t stationName) {
  constexpr std::uint8_t map[15]{0u, 0u, 1u, 1u, 2u, 2u, 3u, 4u,
                                 4u, 5u, 5u, 6u, 6u, 7u, 7u};
  return stationName < 15u ? map[stationName] : 0xFFu;
}
}  // namespace MuonIdBits

/// One hit payload per thread, the device version of makeCudaHitPayloadRow
/// with the HitPayload formulas
__global__ void buildHitPayloadsKernel(CudaMuonSpacePointArrays sp,
                                       CudaUsedHitArrays used,
                                       CudaSurfaceTableArrays surfaces,
                                       CudaHitPayloadArrays out,
                                       std::uint32_t* nMissingSurfaces) {
  const std::uint32_t r{blockIdx.x * blockDim.x + threadIdx.x};
  if (r >= used.nHits) {
    return;
  }
  const std::uint32_t s{used.spacePoint[r]};
  const std::uint32_t raw{sp.muonId[s]};
  const std::uint64_t geometryId{sp.geometryId[s]};
  const std::uint32_t tech{MuonIdBits::technology(raw)};
  const bool straw{tech == MuonIdBits::mdt};
  const bool measEta{MuonIdBits::measuresEta(raw)};
  const bool measPhi{MuonIdBits::measuresPhi(raw)};

  out.positionX[r] = used.positionX[r];
  out.positionY[r] = used.positionY[r];
  out.positionZ[r] = used.positionZ[r];
  out.localPositionY[r] = sp.localPositionY[s];
  out.etaCovariance[r] = sp.covariance1[s];
  out.phiCovariance[r] = sp.covariance0[s];
  out.driftCovariance[r] =
      straw ? Math::square(sp.driftRadius[s]) + sp.covariance1[s] : 0.;
  out.geometryId[r] = geometryId;
  const std::uint32_t stationName{MuonIdBits::stationName(raw)};
  const std::uint32_t side{MuonIdBits::sideA(raw) ? 1u : 0xFFu};
  out.stationKey[r] =
      (side << 24u) | ((stationName & 0xFFu) << 16u) | MuonIdBits::sector(raw);
  out.station[r] = MuonIdBits::station(stationName);
  out.locLayer[r] = static_cast<std::uint8_t>(MuonIdBits::detLayer(raw));
  out.flags[r] = static_cast<std::uint8_t>(
      (straw ? CudaHitFlags::straw : 0u) |
      (measEta ? CudaHitFlags::measuresEta : 0u) |
      (measPhi ? CudaHitFlags::measuresPhi : 0u) |
      (tech == MuonIdBits::mdt || tech == MuonIdBits::mm ||
               (tech == MuonIdBits::sTgc && measEta && !measPhi)
           ? CudaHitFlags::precision
           : 0u));

  // The measurement surface
  std::uint32_t lo{0u};
  std::uint32_t hi{surfaces.nSurfaces};
  while (lo < hi) {
    const std::uint32_t mid{(lo + hi) / 2u};
    if (surfaces.geometryId[mid] < geometryId) {
      lo = mid + 1u;
    } else {
      hi = mid;
    }
  }
  const Math::Vector3 zero{};
  out.sensorDirectionX[r] = 0.;
  out.sensorDirectionY[r] = 0.;
  out.sensorDirectionZ[r] = 0.;
  Math::Vector3 etaDir{zero};
  Math::Vector3 phiDir{zero};
  if (lo == surfaces.nSurfaces || surfaces.geometryId[lo] != geometryId) {
    atomicAdd(nMissingSurfaces, 1u);
  } else {
    const double* rot{surfaces.rotation + 9u * lo};
    // HitPayload: only the eta strips of two-dimensional TGC measurements can
    // be tilted against their phi strips
    bool nonOrthogonal{false};
    double stripAngle{0.};
    if (measEta && measPhi && !straw && tech == MuonIdBits::tgc) {
      const double* bucketRot{used.bucketRotation + 9u * used.bucket[r]};
      const Math::Vector3 etaMeasDir{rotate(bucketRot, sp.toNextSensorX[s],
                                            sp.toNextSensorY[s],
                                            sp.toNextSensorZ[s])};
      const double c{Math::dot(etaMeasDir, column(rot, 1u))};
      if (std::abs(c) > Math::epsilon) {
        stripAngle = std::atan2(Math::dot(etaMeasDir, column(rot, 0u)), c);
        nonOrthogonal = true;
      }
    }
    // HitPayload::globalSensorDirection
    Math::Vector3 sensorDir{};
    if (straw) {
      sensorDir = column(rot, 2u);
    } else if (nonOrthogonal) {
      sensorDir = -std::sin(stripAngle) * column(rot, 1u) +
                  std::cos(stripAngle) * column(rot, 0u);
    } else {
      sensorDir = column(rot, 1u);
    }
    // HitPayload::stripMeasurementDirections, for the hits that measure eta
    if (measEta) {
      out.sensorDirectionX[r] = sensorDir.x;
      out.sensorDirectionY[r] = sensorDir.y;
      out.sensorDirectionZ[r] = sensorDir.z;
      if (!straw) {
        etaDir = measPhi && nonOrthogonal ? cross(sensorDir, column(rot, 2u))
                                          : column(rot, 0u);
        if (measPhi) {
          phiDir = column(rot, 1u);
        }
      }
    }
  }
  out.etaMeasurementDirectionX[r] = etaDir.x;
  out.etaMeasurementDirectionY[r] = etaDir.y;
  out.etaMeasurementDirectionZ[r] = etaDir.z;
  out.phiMeasurementDirectionX[r] = phiDir.x;
  out.phiMeasurementDirectionY[r] = phiDir.y;
  out.phiMeasurementDirectionZ[r] = phiDir.z;
}

/// Copies the patterns of status ok of the event pool to the start of the
/// output records, in order, and their number and the overflow bits. Runs in
/// one block
__global__ void compactEventPatternsKernel(CudaPatternRecordArrays event,
                                           CudaPatternRecordArrays out) {
  __shared__ std::uint32_t warpTotals[32];
  __shared__ std::uint32_t chunkBase;
  const std::uint32_t n{event.nPatterns[0]};
  const std::uint32_t lane{threadIdx.x % lanesPerWarp};
  const std::uint32_t warp{threadIdx.x / lanesPerWarp};
  const std::uint32_t nWarps{blockDim.x / lanesPerWarp};
  if (threadIdx.x == 0u) {
    chunkBase = 0u;
  }
  __syncthreads();
  for (std::uint32_t base = 0u; base < n; base += blockDim.x) {
    const std::uint32_t i{base + threadIdx.x};
    const bool keep{i < n && event.status[i] == CudaPatternStatus::ok};
    const unsigned ballot{__ballot_sync(fullMask, keep)};
    if (lane == 0u) {
      warpTotals[warp] = __popc(ballot);
    }
    __syncthreads();
    std::uint32_t before{chunkBase};
    for (std::uint32_t w = 0u; w < warp; ++w) {
      before += warpTotals[w];
    }
    if (keep) {
      copyRecord(event, i, out, before + __popc(ballot & ((1u << lane) - 1u)));
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
      for (std::uint32_t w = 0u; w < nWarps; ++w) {
        chunkBase += warpTotals[w];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0u) {
    out.nPatterns[0] = chunkBase;
    out.overflow[0] = event.overflow[0];
  }
}

WorkspaceV2 allocateWorkspaceV2(std::size_t nEntries, std::size_t capacity,
                                std::size_t maxHits, cudaStream_t stream) {
  WorkspaceV2 ws{.nEntries = static_cast<std::uint32_t>(nEntries),
                 .capacity = static_cast<std::uint32_t>(capacity),
                 .maxHits = static_cast<std::uint32_t>(maxHits)};
  const std::size_t nWork{nEntries * WarpSeed::nRegions * capacity};
  const std::size_t nScratch{nEntries * capacity};
  auto allocate = [stream](auto*& ptr, std::size_t count) {
    ACTS_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ptr),
                                    count * sizeof(*ptr), stream));
  };
  allocate(ws.patterns, nWork);
  allocate(ws.patternHits, nWork * maxHits);
  allocate(ws.acceptMask, nScratch);
  allocate(ws.lowMask, nScratch);
  allocate(ws.offsets, nScratch);
  allocate(ws.flags, nScratch);
  allocate(ws.state, nScratch);
  allocate(ws.nextState, nScratch);
  allocate(ws.nResults, nEntries);
  allocate(ws.overflow, nEntries);
  allocate(ws.poolOffsets, nEntries);
  return ws;
}

void freeWorkspaceV2(const WorkspaceV2& ws, cudaStream_t stream) {
  for (void* ptr :
       {static_cast<void*>(ws.patterns), static_cast<void*>(ws.patternHits),
        static_cast<void*>(ws.acceptMask), static_cast<void*>(ws.lowMask),
        static_cast<void*>(ws.offsets), static_cast<void*>(ws.flags),
        static_cast<void*>(ws.state), static_cast<void*>(ws.nextState),
        static_cast<void*>(ws.nResults), static_cast<void*>(ws.overflow),
        static_cast<void*>(ws.poolOffsets)}) {
    ACTS_CUDA_CHECK(cudaFreeAsync(ptr, stream));
  }
}

}  // namespace

namespace ActsExamples {

void buildCudaHitPayloads(const CudaMuonSpacePointArrays& spacePoints,
                          const CudaUsedHitArrays& used,
                          const CudaSurfaceTableArrays& surfaces,
                          const CudaHitPayloadArrays& payloads,
                          std::uint32_t* nMissingSurfaces,
                          cudaStream_t stream) {
  if (used.nHits == 0u) {
    return;
  }
  constexpr unsigned blockSize{128u};
  buildHitPayloadsKernel<<<(used.nHits + blockSize - 1u) / blockSize, blockSize,
                           0, stream>>>(spacePoints, used, surfaces, payloads,
                                        nMissingSurfaces);
  ACTS_CUDA_CHECK(cudaGetLastError());
}

CudaPatternRecordContainer findCudaPatternsInEtaV2(
    const CudaHitPayloadContainer& hits, CudaCandidateListContainer& lists,
    const CudaPatternFinderConfig& cfg, const CudaPatternFinderSizes& sizes,
    cudaStream_t stream, CudaPatternFinderTiming* timing) {
  if (sizes.batchSize == 0u || sizes.maxHitsPerPattern == 0u ||
      sizes.maxHitsPerPattern >= cudaInvalidHitPosition) {
    throw std::invalid_argument(
        "findCudaPatternsInEtaV2: the batch size and the hits per pattern have "
        "to be positive, and the hits have to fit the hit positions");
  }
  // The memory pool keeps what is freed between events, see version 1
  {
    int device{0};
    ACTS_CUDA_CHECK(cudaGetDevice(&device));
    cudaMemPool_t pool{};
    ACTS_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, device));
    std::uint64_t keepAll{std::numeric_limits<std::uint64_t>::max()};
    ACTS_CUDA_CHECK(cudaMemPoolSetAttribute(
        pool, cudaMemPoolAttrReleaseThreshold, &keepAll));
  }
  const std::size_t batchSize{
      std::max<std::size_t>(1u, std::min(sizes.batchSize, lists.nSeeds()))};
  const std::size_t capacity{std::max<std::size_t>(
      1u, sizes.workspaceBytes / (batchSize * WorkspaceV2::bytesPerPattern(
                                                  sizes.maxHitsPerPattern)))};

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
  if (!lists.empty()) {
    prepareSeedsKernelV2<<<static_cast<unsigned>(
                               (lists.nSeeds() + warpsPerBlock - 1u) /
                               warpsPerBlock),
                           warpsPerBlock * lanesPerWarp, 0, stream>>>(
        hits.deviceArrays(), lists.deviceArrays(), cfg);
    ACTS_CUDA_CHECK(cudaGetLastError());
  }
  CudaPatternRecordContainer event{CudaPatternRecordContainer::onDevice(
      1u, sizes.maxPatterns, sizes.maxHitsPerPattern, stream)};
  const WorkspaceV2 ws{allocateWorkspaceV2(batchSize, capacity,
                                           sizes.maxHitsPerPattern, stream)};
  const auto nBlocks{
      static_cast<unsigned>((batchSize + warpsPerBlock - 1u) / warpsPerBlock)};
  for (std::size_t firstSeed = 0u; firstSeed < lists.nSeeds();
       firstSeed += batchSize) {
    buildPatternsKernelV2<<<nBlocks, warpsPerBlock * lanesPerWarp, 0, stream>>>(
        hits.deviceArrays(), lists.deviceArrays(), cfg,
        static_cast<std::uint32_t>(firstSeed), ws);
    ACTS_CUDA_CHECK(cudaGetLastError());
    appendPatternsKernel<<<1, 256, 0, stream>>>(
        hits.deviceArrays(), lists.deviceArrays(), cfg,
        static_cast<std::uint32_t>(firstSeed), ws, event.deviceArrays());
    ACTS_CUDA_CHECK(cudaGetLastError());
  }
  freeWorkspaceV2(ws, stream);

  // The links of every pair of the event's patterns, then the rounds
  const std::size_t maxPatterns{sizes.maxPatterns};
  const std::size_t wordsPerRow{(maxPatterns + 31u) / 32u};
  std::uint32_t* good{nullptr};
  std::uint32_t* links{nullptr};
  std::uint8_t* state{nullptr};
  std::uint8_t* nextState{nullptr};
  ACTS_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&good),
                                  maxPatterns * sizeof(std::uint32_t), stream));
  ACTS_CUDA_CHECK(cudaMallocAsync(
      reinterpret_cast<void**>(&links),
      maxPatterns * wordsPerRow * sizeof(std::uint32_t), stream));
  ACTS_CUDA_CHECK(
      cudaMallocAsync(reinterpret_cast<void**>(&state), maxPatterns, stream));
  ACTS_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&nextState),
                                  maxPatterns, stream));
  ACTS_CUDA_CHECK(cudaMemsetAsync(
      links, 0, maxPatterns * wordsPerRow * sizeof(std::uint32_t), stream));
  countEventGroupsKernel<<<static_cast<unsigned>((maxPatterns + 255u) / 256u),
                           256, 0, stream>>>(hits.deviceArrays(), cfg,
                                             event.deviceArrays(), good);
  ACTS_CUDA_CHECK(cudaGetLastError());
  linkEventPatternsKernel<<<512, 256, 0, stream>>>(
      hits.deviceArrays(), cfg, event.deviceArrays(), good, links,
      static_cast<std::uint32_t>(wordsPerRow));
  ACTS_CUDA_CHECK(cudaGetLastError());
  resolveEventOverlapsKernel<<<1, 1024, 0, stream>>>(
      event.deviceArrays(), links, static_cast<std::uint32_t>(wordsPerRow),
      state, nextState);
  ACTS_CUDA_CHECK(cudaGetLastError());
  for (void* ptr : {static_cast<void*>(good), static_cast<void*>(links),
                    static_cast<void*>(state), static_cast<void*>(nextState)}) {
    ACTS_CUDA_CHECK(cudaFreeAsync(ptr, stream));
  }

  // Only the kept patterns go back: compacted on the device, their number
  // read, then their slots copied
  const CudaPatternRecordContainer compacted{
      CudaPatternRecordContainer::onDevice(1u, sizes.maxPatterns,
                                           sizes.maxHitsPerPattern, stream)};
  compactEventPatternsKernel<<<1, 1024, 0, stream>>>(event.deviceArrays(),
                                                     compacted.deviceArrays());
  ACTS_CUDA_CHECK(cudaGetLastError());
  mark(1u);
  std::array<std::uint32_t, 1> nKept{};
  std::array<std::uint8_t, 1> overflow{};
  ACTS_CUDA_CHECK(
      cudaMemcpyAsync(nKept.data(), compacted.deviceArrays().nPatterns,
                      sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
  ACTS_CUDA_CHECK(
      cudaMemcpyAsync(overflow.data(), compacted.deviceArrays().overflow,
                      sizeof(std::uint8_t), cudaMemcpyDeviceToHost, stream));
  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));
  CudaPatternRecordContainer kept{1u, nKept[0], sizes.maxHitsPerPattern};
  kept.copyFromDevice(compacted.deviceArrays(), stream);
  mark(2u);
  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));
  kept.setNPatterns(0u, nKept[0]);
  kept.setOverflow(0u, overflow[0]);

  if (timing != nullptr) {
    auto elapsedUs = [&](std::size_t from, std::size_t to) {
      float ms{0.f};
      ACTS_CUDA_CHECK(cudaEventElapsedTime(&ms, marks[from], marks[to]));
      return 1000. * ms;
    };
    timing->kernelsUs = elapsedUs(0u, 1u);
    timing->downloadUs = elapsedUs(1u, 2u);
    timing->capacityPerSeed = capacity;
    timing->overflow = overflow[0];
    timing->nPatternsInEta = nKept[0];
    for (cudaEvent_t& m : marks) {
      ACTS_CUDA_CHECK(cudaEventDestroy(m));
    }
  }
  return kept;
}

}  // namespace ActsExamples
