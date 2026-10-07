// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFinding/CudaPatternFinder.hpp"
#include "ActsExamples/Utilities/CudaUtilities.hpp"

#include <stdexcept>

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
  if (n == 0u || n < cfg.minLayers) {
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
  if (globLayer[n - 1u] + 1u < cfg.minLayers) {
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

}  // namespace ActsExamples
