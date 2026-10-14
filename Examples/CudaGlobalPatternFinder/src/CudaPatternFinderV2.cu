// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

/// The pattern finding in eta, version 2: a warp builds a seed one layer at a
/// time, with memory taken from one arena as it needs it, and the patterns of
/// the event compete by priority, independently of the seed order. The tests,
/// the residuals, the fits and the cuts are those of version 1
/// (CudaPatternBuilder.cuh); what changes is the order in which they run and
/// the way patterns compete.

#include "ActsExamples/TrackFinding/CudaPatternFinder.hpp"
#include "ActsExamples/Utilities/CudaUtilities.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

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
/// The pattern lost the selection of the patterns that fit and does not
/// carry on
constexpr std::uint8_t flagNoCarry = 1u << 2u;

/// Decision of a pattern in the rounds
constexpr std::uint8_t undecided = 0u;
constexpr std::uint8_t kept = 1u;
constexpr std::uint8_t dropped = 2u;

/// The device memory of an event's pattern building: one block that all seeds
/// take from as they need it, through one counter. Patterns and their scratch
/// are taken below patternLimit; the rest is kept for choosing the best
/// patterns of a layer when the patterns ran out of room
struct Arena {
  char* base{nullptr};
  unsigned long long* used{nullptr};
  unsigned long long size{0u};
  unsigned long long patternLimit{0u};
};

/// Takes `want` bytes below `limit`, in multiples of 8. With partial, takes
/// what is left below the limit when that is less than wanted. Returns the
/// start, and the bytes taken in granted; nullptr and zero if none
__device__ __forceinline__ char* arenaTake(const Arena& arena,
                                           unsigned long long want,
                                           unsigned long long limit,
                                           bool partial,
                                           unsigned long long& granted) {
  want = (want + 7ull) & ~7ull;
  unsigned long long old{atomicAdd(arena.used, 0ull)};
  while (true) {
    granted = 0u;
    if (old >= limit) {
      return nullptr;
    }
    unsigned long long give{want};
    if (old + want > limit) {
      if (!partial) {
        return nullptr;
      }
      give = (limit - old) & ~7ull;
      if (give == 0u) {
        return nullptr;
      }
    }
    const unsigned long long previous{atomicCAS(arena.used, old, old + give)};
    if (previous == old) {
      granted = give;
      return arena.base + old;
    }
    old = previous;
  }
}

/// What a seed leaves for the event: the region holding its patterns, the
/// bytes of one slot and the hits a slot holds, and the slots of its kept
/// patterns
struct SeedResult {
  const char* region{nullptr};
  const std::uint32_t* keptSlots{nullptr};
  std::uint32_t nKept{0u};
  std::uint32_t stride{0u};
  std::uint32_t maxHits{0u};
};

/// Totals of the event over its seeds, read back by the host to size the
/// event's patterns
struct EventCounts {
  unsigned int nPatterns{0u};
  unsigned int maxHits{0u};
  unsigned int largestBeam{0u};
  unsigned int overflow{0u};
};

/// The priority key of a pattern as one sortable number: good groups,
/// bending layers, precision layers, then the mean squared normalised
/// residual, smaller being better. Larger keys come first
__device__ __forceinline__ std::uint64_t priorityKey(std::uint32_t good,
                                                     std::uint32_t bending,
                                                     std::uint32_t precision,
                                                     double meanResidual2) {
  auto clamp = [](std::uint32_t v, std::uint32_t max) {
    return static_cast<std::uint64_t>(v < max ? v : max);
  };
  const float residual{static_cast<float>(meanResidual2)};
  // Positive floats order as their bits; a residual that is not a number
  // ranks last
  const std::uint64_t residualRank{
      residual >= 0.f ? ~static_cast<std::uint64_t>(__float_as_uint(residual)) &
                            0xFFFFFFFFull
                      : 0u};
  return (clamp(good, 0xFu) << 52u) | (clamp(bending, 0xFFu) << 44u) |
         (clamp(precision, 0xFFu) << 36u) | (residualRank << 4u);
}

/// One candidate of the selection: its key and its place in the order the
/// patterns of a layer are written
struct Candidate {
  std::uint64_t key{0u};
  std::uint32_t order{0u};
  std::uint32_t pad{0u};
};

/// One seed built by one warp. Every lane holds this object; the lanes share
/// the patterns and the scratch of the seed, and every loop below runs the
/// same number of times in all of them.
///
/// The patterns live in two regions that the layers swap between. A slot
/// holds a pattern followed by its hit positions, as many as the seed's
/// candidate list has layers: a pattern holds at most one hit per layer, so
/// this cannot overflow. Regions and scratch are taken from the event's
/// arena as the seed needs them and grow by doubling; what a region held
/// before it grew is not needed any more when it grows.
struct WarpSeed {
  SeedBuilder b;
  Arena arena;
  std::uint32_t stride;
  /// The most chunks of 32 hits one layer of the seed has
  std::uint32_t maxChunks;
  Math::Hit* chunkHits;
  std::uint32_t lane;

  /// The two regions, as named fields: an array indexed at run time would
  /// live in local memory
  char* region0{nullptr};
  char* region1{nullptr};
  std::uint32_t regionCap0{0u};
  std::uint32_t regionCap1{0u};
  /// Per pattern of the region being extended: for each chunk of the layer,
  /// the hits it accepts and those of them with low confidence, [pattern]
  /// [chunk]; prefix sums or good groups; the flags; the round decisions
  std::uint32_t scratchCap{0u};
  std::uint32_t* acceptMask{nullptr};
  std::uint32_t* lowMask{nullptr};
  std::uint32_t* offsets{nullptr};
  std::uint8_t* flags{nullptr};
  std::uint8_t* state{nullptr};
  std::uint8_t* nextState{nullptr};
  /// The most patterns the seed held at once
  std::uint32_t largestBeam{0u};

  __device__ __forceinline__ char* regionOf(std::uint32_t r) const {
    return r == 0u ? region0 : region1;
  }
  __device__ __forceinline__ std::uint32_t capOf(std::uint32_t r) const {
    return r == 0u ? regionCap0 : regionCap1;
  }
  __device__ __forceinline__ Pattern& pattern(std::uint32_t r,
                                              std::uint32_t k) {
    return *reinterpret_cast<Pattern*>(regionOf(r) +
                                       static_cast<std::size_t>(k) * stride);
  }

  /// arenaTake by lane 0, the result in every lane
  __device__ __forceinline__ char* take(unsigned long long want,
                                        unsigned long long limit, bool partial,
                                        unsigned long long& granted) {
    unsigned long long start{0u};
    unsigned long long got{0u};
    if (lane == 0u) {
      start = reinterpret_cast<unsigned long long>(
          arenaTake(arena, want, limit, partial, got));
    }
    start = __shfl_sync(fullMask, start, 0);
    granted = __shfl_sync(fullMask, got, 0);
    return reinterpret_cast<char*>(start);
  }

  /// Makes room in region r, whose content is not needed any more, for
  /// `count` patterns, and scratch for as many. The scratch of the first
  /// nLive patterns, the ones being extended, is kept. Returns how many
  /// patterns there is room for, at most count
  __device__ __forceinline__ std::uint32_t reserve(std::uint32_t r,
                                                   std::uint32_t count,
                                                   std::uint32_t nLive) {
    const std::uint32_t cap{capOf(r)};
    if (count <= cap && count <= scratchCap) {
      return count;
    }
    // Grows by doubling, from one warp's worth
    std::uint32_t target{count > 2u * cap ? count : 2u * cap};
    target = target > lanesPerWarp ? target : lanesPerWarp;
    if (scratchCap < target) {
      const unsigned long long maskBytes{4ull * target * maxChunks};
      const unsigned long long bytes{2ull * maskBytes + 4ull * target +
                                     3ull * target};
      unsigned long long got{0u};
      char* block{take(bytes, arena.patternLimit, false, got)};
      if (block != nullptr) {
        auto* newAccept = reinterpret_cast<std::uint32_t*>(block);
        auto* newLow = reinterpret_cast<std::uint32_t*>(block + maskBytes);
        auto* newOffsets =
            reinterpret_cast<std::uint32_t*>(block + 2ull * maskBytes);
        auto* newFlags = reinterpret_cast<std::uint8_t*>(
            block + 2ull * maskBytes + 4ull * target);
        for (std::uint32_t k = lane; k < nLive * maxChunks; k += lanesPerWarp) {
          newAccept[k] = acceptMask[k];
          newLow[k] = lowMask[k];
        }
        for (std::uint32_t k = lane; k < nLive; k += lanesPerWarp) {
          newFlags[k] = flags[k];
        }
        __syncwarp();
        acceptMask = newAccept;
        lowMask = newLow;
        offsets = newOffsets;
        flags = newFlags;
        state = newFlags + target;
        nextState = newFlags + 2u * target;
        scratchCap = target;
      } else {
        target = scratchCap;
      }
    }
    if (capOf(r) < target) {
      unsigned long long got{0u};
      char* block{take(static_cast<unsigned long long>(target) * stride,
                       arena.patternLimit, true, got)};
      const auto grown = static_cast<std::uint32_t>(got / stride);
      if (grown > capOf(r)) {
        (r == 0u ? region0 : region1) = block;
        (r == 0u ? regionCap0 : regionCap1) = grown;
      }
    }
    const std::uint32_t room{capOf(r) < scratchCap ? capOf(r) : scratchCap};
    return room < count ? room : count;
  }

  /// The priority key of the pattern a candidate would give: pattern i of
  /// region `from` with the hit at position pos of the candidate list
  __device__ __forceinline__ std::uint64_t childKey(Pattern& parent,
                                                    std::uint16_t pos) {
    const LineTest res{b.checkLine(parent, pos)};
    std::uint32_t counts[Math::Station::nGroups]{};
    const std::uint16_t* hitPos{b.hitsOf(parent)};
    for (std::uint16_t k = 0u; k < parent.nHits; ++k) {
      ++counts[b.group(hitPos[k])];
    }
    ++counts[b.group(pos)];
    std::uint32_t good{0u};
    for (std::uint32_t g = 0u; g < Math::Station::nGroups; ++g) {
      good += counts[g] > 0u && counts[g] >= b.cfg.minGroupLayers;
    }
    return priorityKey(
        good, parent.nBendingLayers() + 1u,
        parent.nPrecisionLayers + (b.isPrecision(pos) ? 1u : 0u),
        (parent.meanNormResidual2 + Math::square(res.residual / res.sigma)) /
            (parent.nDoF + res.nDof));
  }

  /// When the patterns of a layer do not all fit: keeps the `room` best of
  /// the candidates, by the priority key, and takes the others out of the
  /// masks and the carries. A candidate is a would-be pattern, a pattern with
  /// one hit it accepted or a pattern carrying on. The order of a candidate
  /// is its place in the writing order, which breaks ties
  __device__ __forceinline__ void selectBest(
      std::uint32_t from, std::uint32_t n, std::uint32_t first,
      std::uint32_t nChunks, std::uint32_t total, std::uint32_t room) {
    auto carries = [&](std::uint32_t i) {
      return (flags[i] & (flagDropped | flagNormalHit | flagNoCarry)) == 0u;
    };
    const std::uint32_t carryOrder{nChunks * n * 32u};
    auto drop = [&](std::uint32_t order) {
      if (order >= carryOrder) {
        flags[order - carryOrder] |= flagNoCarry;
        return;
      }
      const std::uint32_t pair{order / 32u};
      const std::uint32_t c{pair / n};
      const std::uint32_t i{pair % n};
      atomicAnd(&acceptMask[i * maxChunks + c], ~(1u << (order % 32u)));
    };
    unsigned long long got{0u};
    char* block{take(8ull + 16ull * total, arena.size, false, got)};
    if (block == nullptr) {
      // Not even room to rank them: the candidates written last are dropped
      std::uint32_t nTaken{0u};
      if (lane == 0u) {
        for (std::uint32_t c = 0u; c < nChunks; ++c) {
          for (std::uint32_t i = 0u; i < n; ++i) {
            for (std::uint32_t bit = 0u; bit < 32u; ++bit) {
              if (((acceptMask[i * maxChunks + c] >> bit) & 1u) != 0u &&
                  nTaken++ >= room) {
                drop((c * n + i) * 32u + bit);
              }
            }
          }
        }
        for (std::uint32_t i = 0u; i < n; ++i) {
          if (carries(i) && nTaken++ >= room) {
            drop(carryOrder + i);
          }
        }
      }
      __syncwarp();
      return;
    }
    auto* nWritten = reinterpret_cast<unsigned int*>(block);
    auto* candidates = reinterpret_cast<Candidate*>(block + 8);
    if (lane == 0u) {
      *nWritten = 0u;
    }
    __syncwarp();
    for (std::uint32_t k = lane; k < n * nChunks; k += lanesPerWarp) {
      const std::uint32_t i{k / nChunks};
      const std::uint32_t c{k % nChunks};
      std::uint32_t mask{acceptMask[i * maxChunks + c]};
      while (mask != 0u) {
        const auto bit = static_cast<std::uint32_t>(__ffs(mask) - 1);
        mask &= mask - 1u;
        const auto pos = static_cast<std::uint16_t>(first + 32u * c + bit);
        candidates[atomicAdd(nWritten, 1u)] =
            Candidate{.key = childKey(pattern(from, i), pos),
                      .order = (c * n + i) * 32u + bit};
      }
    }
    for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
      if (carries(i)) {
        const Pattern& pat{pattern(from, i)};
        candidates[atomicAdd(nWritten, 1u)] = Candidate{
            .key = priorityKey(b.countGroups(pattern(from, i), true),
                               pat.nBendingLayers(), pat.nPrecisionLayers,
                               pat.meanResidual2()),
            .order = carryOrder + i};
      }
    }
    __syncwarp();
    // Best first: the larger key, then the earlier order. The network only
    // ever puts the better element first, so the padding is never touched
    auto better = [](const Candidate& a, const Candidate& c) {
      return a.key > c.key || (a.key == c.key && a.order < c.order);
    };
    std::uint32_t padded{1u};
    while (padded < total) {
      padded <<= 1u;
    }
    for (std::uint32_t k = 2u; k <= padded; k <<= 1u) {
      for (std::uint32_t j = k / 2u; j > 0u; j >>= 1u) {
        for (std::uint32_t t = lane; t < padded; t += lanesPerWarp) {
          const std::uint32_t partner{j == k / 2u ? t ^ (k - 1u) : t ^ j};
          if (partner <= t || partner >= total) {
            continue;
          }
          if (better(candidates[partner], candidates[t])) {
            const Candidate swap{candidates[t]};
            candidates[t] = candidates[partner];
            candidates[partner] = swap;
          }
        }
        __syncwarp();
      }
    }
    for (std::uint32_t r = room + lane; r < total; r += lanesPerWarp) {
      drop(candidates[r].order);
    }
    __syncwarp();
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
  ///  3. room: the patterns of the layer are counted before any is written,
  ///     and when they do not all fit the best are kept (selectBest);
  ///  4. rebuild: each compatible hit gives a child, the pattern with the
  ///     hit. A pattern carries on as it is unless it took a hit of normal
  ///     confidence.
  /// A pattern of region `from` marked isOverlap takes no part.
  __device__ __forceinline__ std::uint32_t extendLayer(std::uint32_t from,
                                                       std::uint32_t n,
                                                       std::uint32_t to,
                                                       std::uint32_t first,
                                                       std::uint32_t last) {
    const auto layerPos = static_cast<std::uint16_t>(first);
    const int layer{b.globLayer(layerPos)};
    const std::uint8_t layerGroup{b.group(layerPos)};
    const std::uint32_t nChunks{(last - first + lanesPerWarp - 1u) /
                                lanesPerWarp};
    auto missedLayers = [&](const Pattern& pat) {
      const int d{static_cast<int>(b.globLayer(pat.lastInsertedHit)) - layer};
      return static_cast<std::uint32_t>(d < 0 ? -d : d);
    };

    // 1. Prune
    for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
      const Pattern& pat{pattern(from, i)};
      if (pat.isOverlap) {
        flags[i] = flagDropped;
        continue;
      }
      std::uint32_t minMissed{UINT_MAX};
      std::uint32_t nFamily{0u};
      bool anyLong{false};
      for (std::uint32_t j = 0u; j < n; ++j) {
        const Pattern& other{pattern(from, j)};
        if (other.family != pat.family || other.isOverlap) {
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
          const Pattern& other{pattern(from, j)};
          drop = j != i && !other.isOverlap && other.family == pat.family &&
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
        b.updateLineParameters(pattern(from, i));
      }
    }
    __syncwarp();

    // 2. Test, 32 hits at a time
    auto loadChunk = [&](std::uint32_t c) {
      const std::uint32_t chunk{first + c * lanesPerWarp};
      const std::uint32_t m{last - chunk < lanesPerWarp ? last - chunk
                                                        : lanesPerWarp};
      if (lane < m) {
        chunkHits[lane] = b.hit(static_cast<std::uint16_t>(chunk + lane));
      }
      __syncwarp();
      return m;
    };
    for (std::uint32_t c = 0u; c < nChunks; ++c) {
      for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
        acceptMask[i * maxChunks + c] = 0u;
        lowMask[i * maxChunks + c] = 0u;
      }
      const std::uint32_t m{loadChunk(c)};
      const std::uint32_t chunk{first + c * lanesPerWarp};
      for (std::uint32_t k = lane; k < n * m; k += lanesPerWarp) {
        const std::uint32_t i{k / m};
        const std::uint32_t h{k % m};
        if ((flags[i] & flagDropped) != 0u) {
          continue;
        }
        const LineTest res{b.checkLine(pattern(from, i),
                                       static_cast<std::uint16_t>(chunk + h),
                                       chunkHits[h])};
        if (res.decision != Decision::addHit) {
          continue;
        }
        atomicOr(&acceptMask[i * maxChunks + c], 1u << h);
        if (res.sigma > b.cfg.lowConfidenceResSigma &&
            res.residual / res.sigma > 2.) {
          atomicOr(&lowMask[i * maxChunks + c], 1u << h);
        }
      }
      __syncwarp();
    }
    for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
      for (std::uint32_t c = 0u; c < nChunks; ++c) {
        if ((acceptMask[i * maxChunks + c] & ~lowMask[i * maxChunks + c]) !=
            0u) {
          flags[i] |= flagNormalHit;
        }
      }
    }
    __syncwarp();

    // 3. Room for all of them, or the best that fit
    auto carries = [&](std::uint32_t i) {
      return (flags[i] & (flagDropped | flagNormalHit | flagNoCarry)) == 0u;
    };
    auto nChildrenOf = [&](std::uint32_t i) {
      std::uint32_t count{0u};
      for (std::uint32_t c = 0u; c < nChunks; ++c) {
        count += __popc(acceptMask[i * maxChunks + c]);
      }
      return count;
    };
    const std::uint32_t total{
        warpExclusiveScan(n, nChildrenOf, offsets, lane) +
        warpExclusiveScan(
            n, [&](std::uint32_t i) { return carries(i) ? 1u : 0u; }, offsets,
            lane)};
    const std::uint32_t room{reserve(to, total, n)};
    if (room < total) {
      b.overflow |= CudaPatternOverflow::patterns;
      selectBest(from, n, first, nChunks, total, room);
    }

    // 4. Rebuild, chunk after chunk, then the patterns that carry on
    std::uint32_t nOut{0u};
    for (std::uint32_t c = 0u; c < nChunks; ++c) {
      loadChunk(c);
      const std::uint32_t chunk{first + c * lanesPerWarp};
      const std::uint32_t nChildren{warpExclusiveScan(
          n,
          [&](std::uint32_t i) {
            return __popc(acceptMask[i * maxChunks + c]);
          },
          offsets, lane)};
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
        const std::uint32_t h{
            nthSetBit(acceptMask[lo * maxChunks + c], t - offsets[lo])};
        const auto test = static_cast<std::uint16_t>(chunk + h);
        Pattern& child{pattern(to, nOut + t)};
        b.copy(pattern(from, lo), child);
        b.addHit(child, test, b.checkLine(child, test, chunkHits[h]));
      }
      nOut += nChildren;
      __syncwarp();
    }
    const std::uint32_t nCarried{warpExclusiveScan(
        n, [&](std::uint32_t i) { return carries(i) ? 1u : 0u; }, offsets,
        lane)};
    for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
      if (carries(i)) {
        b.copy(pattern(from, i), pattern(to, nOut + offsets[i]));
      }
    }
    __syncwarp();
    nOut += nCarried;
    largestBeam = nOut > largestBeam ? nOut : largestBeam;
    return nOut;
  }

  /// The overlaps among the n patterns of a region, by priority, in place: in
  /// each round, a pattern that no undecided or kept pattern above it
  /// overlaps is kept, and one that a kept pattern above it overlaps is
  /// dropped. With perFamily, only patterns of the same family compete. The
  /// dropped patterns are marked isOverlap; a pattern marked already takes no
  /// part
  __device__ __forceinline__ void resolveOverlaps(std::uint32_t r,
                                                  std::uint32_t n,
                                                  bool perFamily) {
    std::uint32_t* good{offsets};
    for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
      const Pattern& pat{pattern(r, i)};
      state[i] = pat.isOverlap ? dropped : undecided;
      good[i] = pat.isOverlap ? 0u : b.countGroups(pattern(r, i), true);
    }
    __syncwarp();
    while (true) {
      bool anyUndecided{false};
      for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
        nextState[i] = state[i];
        if (state[i] != undecided) {
          continue;
        }
        const Pattern& pat{pattern(r, i)};
        const PatternInfo info{b.info(pat)};
        bool isDropped{false};
        bool isBlocked{false};
        for (std::uint32_t j = 0u; j < n && !isDropped; ++j) {
          const Pattern& other{pattern(r, j)};
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
    for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
      if (state[i] == dropped) {
        pattern(r, i).isOverlap = true;
      }
    }
    __syncwarp();
  }

  /// Builds the patterns of a seed. Returns where its kept patterns are
  __device__ __forceinline__ SeedResult build(std::uint16_t seedPos,
                                              std::uint32_t nCands,
                                              std::int8_t sector) {
    if (reserve(0u, 1u, 0u) == 0u) {
      b.overflow |= CudaPatternOverflow::patterns;
      return SeedResult{};
    }
    if (lane == 0u) {
      b.makeSeedPattern(pattern(0u, 0u), seedPos, sector);
    }
    __syncwarp();
    const std::uint16_t seedLayer{b.globLayer(seedPos)};
    std::uint32_t current{0u};
    std::uint32_t n{1u};

    // Forward: the layers after the seed layer, outwards
    std::uint32_t first{seedPos + 1u};
    while (first < nCands && b.globLayer(first) == seedLayer) {
      ++first;
    }
    while (first < nCands && n > 0u) {
      std::uint32_t last{first + 1u};
      while (last < nCands && b.globLayer(last) == b.globLayer(first)) {
        ++last;
      }
      n = extendLayer(current, n, 1u - current, first, last);
      current = 1u - current;
      first = last;
    }
    if (n > 1u) {
      resolveOverlaps(current, n, false);
    }

    // Backward: every forward pattern from the seed inwards, side by side.
    // Each is a family of its own
    for (std::uint32_t f = lane; f < n; f += lanesPerWarp) {
      Pattern& start{pattern(current, f)};
      if (start.isOverlap) {
        continue;
      }
      b.moveLineAnchorHit(start, seedPos);
      start.lastInsertedHit = seedPos;
      start.family = static_cast<std::uint16_t>(f);
    }
    __syncwarp();
    std::uint32_t last{seedPos};
    while (last > 0u && b.globLayer(last - 1u) == seedLayer) {
      --last;
    }
    while (last > 0u && n > 0u) {
      std::uint32_t begin{last - 1u};
      while (begin > 0u && b.globLayer(begin - 1u) == b.globLayer(last - 1u)) {
        --begin;
      }
      n = extendLayer(current, n, 1u - current, begin, last);
      current = 1u - current;
      last = begin;
    }

    // The overlaps inside each family, as at the end of each backward pass of
    // the CPU, then among all of them
    if (n > 1u) {
      resolveOverlaps(current, n, true);
      resolveOverlaps(current, n, false);
    }

    // Finalize, keep the patterns that pass the cuts and list them
    for (std::uint32_t k = lane; k < n; k += lanesPerWarp) {
      Pattern& pat{pattern(current, k)};
      if (!pat.isOverlap) {
        pat.meanNormResidual2 /= pat.nDoF;
        pat.isOverlap = !b.passPatternCuts(pat);
      }
    }
    __syncwarp();
    const std::uint32_t nKept{warpExclusiveScan(
        n,
        [&](std::uint32_t k) {
          return pattern(current, k).isOverlap ? 0u : 1u;
        },
        offsets, lane)};
    std::uint32_t* list{lowMask};
    for (std::uint32_t k = lane; k < n; k += lanesPerWarp) {
      if (!pattern(current, k).isOverlap) {
        list[offsets[k]] = k;
      }
    }
    __syncwarp();
    return SeedResult{.region = regionOf(current),
                      .keptSlots = list,
                      .nKept = nKept,
                      .stride = stride,
                      .maxHits = b.maxHits};
  }
};

/// The builder of a seed whose patterns hold maxHits hits in slots of
/// `stride` bytes
__device__ SeedBuilder seedBuilder(const CudaHitPayloadArrays& hits,
                                   const CudaCandidateArrays& lists,
                                   const CudaPatternFinderConfig& cfg,
                                   std::uint32_t seed, std::uint32_t maxHits,
                                   std::uint32_t stride) {
  return SeedBuilder{.hits = hits,
                     .cfg = cfg,
                     .rows = lists.candidateIndices + lists.begin(seed),
                     .globLayers = lists.candidateGlobLayer + lists.begin(seed),
                     .patterns = nullptr,
                     .patternHits = nullptr,
                     .capacity = 0u,
                     .maxHits = maxHits,
                     .slotStride = stride};
}

/// Builds every seed of the event, one warp per seed, with memory from the
/// arena
__global__ void buildPatternsKernelV2(CudaHitPayloadArrays hits,
                                      CudaCandidateArrays lists,
                                      CudaPatternFinderConfig cfg, Arena arena,
                                      SeedResult* results,
                                      EventCounts* counts) {
  const std::uint32_t seed{(blockIdx.x * blockDim.x + threadIdx.x) /
                           lanesPerWarp};
  const std::uint32_t lane{threadIdx.x % lanesPerWarp};
  if (seed >= lists.nSeeds) {
    return;
  }
  if (lane == 0u) {
    results[seed] = SeedResult{};
  }
  if (lists.seedStatus[seed] != CudaSeedStatus::accepted ||
      lists.seedPosition[seed] == cudaInvalidHitPosition) {
    return;
  }
  // The hits of a pattern: one per layer of the list. The chunks: the most
  // hits one layer has, in chunks of 32
  const std::uint32_t nCands{lists.end(seed) - lists.begin(seed)};
  const std::uint16_t* globLayers{lists.candidateGlobLayer + lists.begin(seed)};
  const auto maxHits = static_cast<std::uint32_t>(globLayers[nCands - 1u] + 1u);
  std::uint32_t longestLayer{0u};
  for (std::uint32_t k = 0u, run = 0u; k < nCands; ++k) {
    run = (k > 0u && globLayers[k] == globLayers[k - 1u]) ? run + 1u : 1u;
    longestLayer = run > longestLayer ? run : longestLayer;
  }
  const std::uint32_t stride{static_cast<std::uint32_t>(
      (sizeof(Pattern) + maxHits * sizeof(std::uint16_t) + 7u) & ~7u)};

  // The hits of the chunk being tested, per warp of the block. Raw storage,
  // since shared memory takes no constructors
  __shared__ alignas(Math::Hit) unsigned char
      chunkStorage[warpsPerBlock * lanesPerWarp * sizeof(Math::Hit)];
  Math::Hit* chunkHits{reinterpret_cast<Math::Hit*>(chunkStorage) +
                       (threadIdx.x / lanesPerWarp) * lanesPerWarp};
  WarpSeed warp{.b = seedBuilder(hits, lists, cfg, seed, maxHits, stride),
                .arena = arena,
                .stride = stride,
                .maxChunks = (longestLayer + lanesPerWarp - 1u) / lanesPerWarp,
                .chunkHits = chunkHits,
                .lane = lane};
  const SeedResult result{
      warp.build(lists.seedPosition[seed], nCands,
                 static_cast<std::int8_t>(lists.seedSector[seed]))};
  const unsigned overflow{
      __reduce_or_sync(fullMask, static_cast<unsigned>(warp.b.overflow))};
  if (lane == 0u) {
    results[seed] = result;
    atomicAdd(&counts->nPatterns, result.nKept);
    if (result.nKept > 0u) {
      atomicMax(&counts->maxHits, maxHits);
    }
    atomicMax(&counts->largestBeam, warp.largestBeam);
    if (overflow != 0u) {
      atomicOr(&counts->overflow, overflow);
    }
  }
}

/// The memory of the event's patterns, taken from the arena on the device
/// once the seeds are built, so that the host does not wait for their
/// number: the patterns of every seed, the kept ones, the good groups, the
/// links of every pair, the round decisions and where each seed's patterns
/// start. nKept is written by the compaction
struct EventBuffers {
  CudaPatternRecordArrays event{};
  CudaPatternRecordArrays compacted{};
  std::uint32_t* good{nullptr};
  std::uint32_t* links{nullptr};
  std::uint8_t* state{nullptr};
  std::uint8_t* nextState{nullptr};
  std::uint32_t* firstOfSeed{nullptr};
  std::uint32_t nPatterns{0u};
  std::uint32_t wordsPerRow{0u};
  std::uint32_t nKept{0u};
};

/// Points the columns of pattern records with n slots of maxHits hits into
/// memory from base on, or only counts their bytes if base is nullptr
__device__ void carveRecords(CudaPatternRecordArrays& r, char* base,
                             unsigned long long& cursor, std::uint32_t n,
                             std::uint32_t maxHits) {
  auto take = [&](auto*& column, unsigned long long count) {
    if (base != nullptr) {
      column = reinterpret_cast<std::remove_reference_t<decltype(column)>>(
          base + cursor);
    }
    cursor += (count * sizeof(*column) + 15ull) & ~15ull;
  };
  const unsigned long long hitSlots{static_cast<unsigned long long>(n) *
                                    maxHits};
  take(r.sector, n);
  take(r.patPhi, n);
  take(r.patTheta, n);
  take(r.patPhiOffset, n);
  take(r.bendPlaneCovPhiPhi, n);
  take(r.bendPlaneCovSS, n);
  take(r.bendPlaneCovPhiS, n);
  take(r.nDoF, n);
  take(r.lastNDoF, n);
  take(r.anchorRadius, n);
  take(r.meanNormResidual2, n);
  take(r.lastResidual, n);
  take(r.lastResSigma, n);
  take(r.linePosX, n);
  take(r.linePosY, n);
  take(r.linePosZ, n);
  take(r.lineDirX, n);
  take(r.lineDirY, n);
  take(r.lineDirZ, n);
  take(r.leverArm, n);
  take(r.bendPlaneNormX, n);
  take(r.bendPlaneNormY, n);
  take(r.bendPlaneNormZ, n);
  take(r.nPrecisionLayers, n);
  take(r.nTriggerLayers, n);
  take(r.nPhiLayers, n);
  take(r.nHits, n);
  take(r.lastInsertedHit, n);
  take(r.prevLayerHit, n);
  take(r.lineAnchorHit, n);
  take(r.flags, n);
  take(r.status, n);
  take(r.hitIndex, hitSlots);
  take(r.globLayer, hitSlots);
  take(r.nPatterns, 1u);
  take(r.overflow, 1u);
  r.nSeeds = 1u;
  r.capacityPerSeed = n;
  r.maxHitsPerPattern = maxHits;
}

/// Takes the event's buffers from the arena, sized by the totals of the
/// seeds. When the arena has not room for them, the event keeps no patterns
/// and gets the eventPatterns bit. Runs in one thread
__global__ void carveEventKernel(Arena arena, EventCounts* counts,
                                 std::uint32_t nSeeds, EventBuffers* buffers) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) {
    return;
  }
  EventBuffers out{};
  const std::uint32_t n{counts->nPatterns};
  const std::uint32_t maxHits{counts->maxHits > 0u ? counts->maxHits : 1u};
  const std::uint32_t words{(n + 31u) / 32u};
  auto layout = [&](char* base) {
    unsigned long long cursor{0u};
    carveRecords(out.event, base, cursor, n, maxHits);
    carveRecords(out.compacted, base, cursor, n, maxHits);
    auto take = [&](auto*& column, unsigned long long count) {
      if (base != nullptr) {
        column = reinterpret_cast<std::remove_reference_t<decltype(column)>>(
            base + cursor);
      }
      cursor += (count * sizeof(*column) + 15ull) & ~15ull;
    };
    take(out.good, n);
    take(out.links, static_cast<unsigned long long>(n) * words);
    take(out.state, n);
    take(out.nextState, n);
    take(out.firstOfSeed, nSeeds);
    return cursor;
  };
  if (n > 0u) {
    unsigned long long granted{0u};
    char* base{arenaTake(arena, layout(nullptr), arena.size, false, granted)};
    if (base != nullptr) {
      layout(base);
      out.nPatterns = n;
      out.wordsPerRow = words;
    } else {
      counts->overflow |= CudaPatternOverflow::eventPatterns;
    }
  }
  *buffers = out;
}

/// Copies the kept patterns of every seed to the event's records, in seed
/// order, which are sized to hold them all. Runs in one block
__global__ void appendPatternsKernel(CudaHitPayloadArrays hits,
                                     CudaCandidateArrays lists,
                                     CudaPatternFinderConfig cfg,
                                     const SeedResult* results,
                                     const EventCounts* counts,
                                     const EventBuffers* buffers) {
  if (buffers->nPatterns == 0u) {
    return;
  }
  const CudaPatternRecordArrays event{buffers->event};
  std::uint32_t* firstOfSeed{buffers->firstOfSeed};
  __shared__ std::uint32_t nTotal;
  if (threadIdx.x == 0u) {
    std::uint32_t sum{0u};
    for (std::uint32_t s = 0u; s < lists.nSeeds; ++s) {
      firstOfSeed[s] = sum;
      sum += results[s].nKept;
    }
    nTotal = sum;
    event.nPatterns[0] = sum;
    event.overflow[0] = static_cast<std::uint8_t>(counts->overflow);
  }
  __syncthreads();
  for (std::uint32_t t = threadIdx.x; t < nTotal; t += blockDim.x) {
    std::uint32_t lo{0u};
    std::uint32_t hi{lists.nSeeds};
    while (hi - lo > 1u) {
      const std::uint32_t mid{(lo + hi) / 2u};
      if (firstOfSeed[mid] <= t) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const SeedResult& result{results[lo]};
    SeedBuilder builder{
        seedBuilder(hits, lists, cfg, lo, result.maxHits, result.stride)};
    const auto& pat = *reinterpret_cast<const Pattern*>(
        result.region +
        static_cast<std::size_t>(result.keptSlots[t - firstOfSeed[lo]]) *
            result.stride);
    writePattern(builder, pat, builder.hitsOf(pat), event, t);
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
                                        const EventBuffers* buffers) {
  const std::uint32_t n{buffers->nPatterns};
  const CudaPatternRecordArrays event{buffers->event};
  const std::uint32_t* good{buffers->good};
  std::uint32_t* links{buffers->links};
  const std::uint32_t wordsPerRow{buffers->wordsPerRow};
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

/// The good groups of each pattern of the event, and the links zeroed. The
/// threads of the grid stride over them
__global__ void countEventGroupsKernel(CudaHitPayloadArrays hits,
                                       CudaPatternFinderConfig cfg,
                                       const EventBuffers* buffers) {
  const std::uint32_t n{buffers->nPatterns};
  const CudaPatternRecordArrays event{buffers->event};
  const std::uint32_t stride{gridDim.x * blockDim.x};
  const std::uint32_t first{blockIdx.x * blockDim.x + threadIdx.x};
  const unsigned long long nWords{static_cast<unsigned long long>(n) *
                                  buffers->wordsPerRow};
  for (unsigned long long w = first; w < nWords; w += stride) {
    buffers->links[w] = 0u;
  }
  for (std::uint32_t i = first; i < n; i += stride) {
    buffers->good[i] = countGroups(
        hits, event.nHits[i],
        [&event, i](std::uint32_t k) {
          return event.hitIndex[event.hitSlot(i, k)];
        },
        cfg.minGroupLayers);
  }
}

/// The overlaps among the patterns of the event, by priority, as
/// WarpSeed::resolveOverlaps does it for a seed, on the links. The dropped
/// patterns get the status rejected. Runs in one block
__global__ void resolveEventOverlapsKernel(const EventBuffers* buffers) {
  const std::uint32_t n{buffers->nPatterns};
  const CudaPatternRecordArrays event{buffers->event};
  const std::uint32_t* links{buffers->links};
  const std::uint32_t wordsPerRow{buffers->wordsPerRow};
  std::uint8_t* state{buffers->state};
  std::uint8_t* nextState{buffers->nextState};
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
  // Row r of the payloads is space point r of the container, which holds the
  // used hits only
  const std::uint32_t s{r};
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
      // The bucket of the hit: the last one that starts at or before it
      std::uint32_t bucket{0u};
      std::uint32_t pastBucket{used.nBuckets};
      while (pastBucket - bucket > 1u) {
        const std::uint32_t mid{(bucket + pastBucket) / 2u};
        if (sp.bucketStart[mid] <= r) {
          bucket = mid;
        } else {
          pastBucket = mid;
        }
      }
      const double* bucketRot{used.bucketRotation + 9u * bucket};
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
__global__ void compactEventPatternsKernel(EventBuffers* buffers) {
  __shared__ std::uint32_t warpTotals[32];
  __shared__ std::uint32_t chunkBase;
  const std::uint32_t n{buffers->nPatterns};
  if (n == 0u) {
    return;
  }
  const CudaPatternRecordArrays event{buffers->event};
  const CudaPatternRecordArrays out{buffers->compacted};
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
    buffers->nKept = chunkBase;
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
  if (sizes.workspaceBytes == 0u) {
    throw std::invalid_argument(
        "findCudaPatternsInEtaV2: the workspace has to have some memory");
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
  // Device memory of the event, released once its work is done
  std::vector<void*> buffers{};
  auto allocate = [&](auto*& ptr, std::size_t count) {
    ACTS_CUDA_CHECK(cudaMallocAsync(
        reinterpret_cast<void**>(&ptr),
        std::max<std::size_t>(count, 1u) * sizeof(*ptr), stream));
    buffers.push_back(ptr);
  };
  auto release = [&]() {
    for (void* ptr : buffers) {
      ACTS_CUDA_CHECK(cudaFreeAsync(ptr, stream));
    }
    buffers.clear();
  };
  const std::uint32_t nSeeds{static_cast<std::uint32_t>(lists.nSeeds())};
  const unsigned nWarpBlocks{(nSeeds + warpsPerBlock - 1u) / warpsPerBlock};

  mark(0u);
  // The seed checks
  if (nSeeds > 0u) {
    prepareSeedsKernelV2<<<nWarpBlocks, warpsPerBlock * lanesPerWarp, 0,
                           stream>>>(hits.deviceArrays(), lists.deviceArrays(),
                                     cfg);
    ACTS_CUDA_CHECK(cudaGetLastError());
  }

  // Every seed at once, with the memory of the arena; then the totals, which
  // size the event's patterns
  Arena arena{};
  arena.size = sizes.workspaceBytes & ~7ull;
  // A sixteenth is kept for choosing the best patterns when they run short
  arena.patternLimit = arena.size - arena.size / 16u;
  allocate(arena.base, arena.size);
  allocate(arena.used, 1u);
  ACTS_CUDA_CHECK(
      cudaMemsetAsync(arena.used, 0, sizeof(unsigned long long), stream));
  SeedResult* results{nullptr};
  EventCounts* counts{nullptr};
  allocate(results, nSeeds);
  allocate(counts, 1u);
  ACTS_CUDA_CHECK(cudaMemsetAsync(counts, 0, sizeof(EventCounts), stream));
  if (nSeeds > 0u) {
    buildPatternsKernelV2<<<nWarpBlocks, warpsPerBlock * lanesPerWarp, 0,
                            stream>>>(hits.deviceArrays(), lists.deviceArrays(),
                                      cfg, arena, results, counts);
    ACTS_CUDA_CHECK(cudaGetLastError());
  }
  // The event's buffers, sized on the device by what the seeds produced,
  // then the patterns of every seed, the links of every pair and the rounds
  EventBuffers* eventBuffers{nullptr};
  allocate(eventBuffers, 1u);
  carveEventKernel<<<1, 1, 0, stream>>>(arena, counts, nSeeds, eventBuffers);
  ACTS_CUDA_CHECK(cudaGetLastError());
  appendPatternsKernel<<<1, 256, 0, stream>>>(hits.deviceArrays(),
                                              lists.deviceArrays(), cfg,
                                              results, counts, eventBuffers);
  ACTS_CUDA_CHECK(cudaGetLastError());
  countEventGroupsKernel<<<128, 256, 0, stream>>>(hits.deviceArrays(), cfg,
                                                  eventBuffers);
  ACTS_CUDA_CHECK(cudaGetLastError());
  linkEventPatternsKernel<<<512, 256, 0, stream>>>(hits.deviceArrays(), cfg,
                                                   eventBuffers);
  ACTS_CUDA_CHECK(cudaGetLastError());
  resolveEventOverlapsKernel<<<1, 1024, 0, stream>>>(eventBuffers);
  ACTS_CUDA_CHECK(cudaGetLastError());
  // Only the kept patterns go back: compacted on the device, their number
  // read, then their slots copied
  compactEventPatternsKernel<<<1, 256, 0, stream>>>(eventBuffers);
  ACTS_CUDA_CHECK(cudaGetLastError());
  mark(1u);
  EventBuffers ready{};
  EventCounts totals{};
  ACTS_CUDA_CHECK(cudaMemcpyAsync(&ready, eventBuffers, sizeof(EventBuffers),
                                  cudaMemcpyDeviceToHost, stream));
  ACTS_CUDA_CHECK(cudaMemcpyAsync(&totals, counts, sizeof(EventCounts),
                                  cudaMemcpyDeviceToHost, stream));
  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));
  const std::uint32_t maxHits{std::max(totals.maxHits, 1u)};
  CudaPatternRecordContainer kept{1u, ready.nKept, maxHits};
  if (ready.nKept > 0u) {
    kept.copyFromDevice(ready.compacted, stream);
  }
  mark(2u);
  release();
  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));
  const auto overflow = static_cast<std::uint8_t>(totals.overflow);
  const std::array<std::uint32_t, 1> nKept{ready.nKept};
  kept.setNPatterns(0u, nKept[0]);
  kept.setOverflow(0u, overflow);

  if (timing != nullptr) {
    auto elapsedUs = [&](std::size_t from, std::size_t to) {
      float ms{0.f};
      ACTS_CUDA_CHECK(cudaEventElapsedTime(&ms, marks[from], marks[to]));
      return 1000. * ms;
    };
    timing->kernelsUs = elapsedUs(0u, 1u);
    timing->downloadUs = elapsedUs(1u, 2u);
    timing->capacityPerSeed = totals.largestBeam;
    timing->overflow = overflow;
    timing->nPatternsInEta = nKept[0];
    for (cudaEvent_t& m : marks) {
      ACTS_CUDA_CHECK(cudaEventDestroy(m));
    }
  }
  return kept;
}

}  // namespace ActsExamples
