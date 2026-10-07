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

/// One pattern while it is built: PatternState with its hits, the last
/// inserted hit, the last hit of the previous layer and the line anchor held
/// as positions in the seed's sorted candidate list. A position names both
/// the hit and its global layer, as the CPU's OrderedHit does
struct Pattern {
  Math::Vector3 linePos{};
  Math::Vector3 lineDir{};
  Math::Vector3 bendPlaneNorm{};
  double leverArm{0.};
  double meanNormResidual2{0.};
  double lastResidual{0.};
  double lastResSigma{0.};
  double patPhi{0.};
  double patTheta{0.};
  double patPhiCov{0.};
  std::uint16_t lastInsertedHit{0u};
  std::uint16_t prevLayerHit{0u};
  std::uint16_t lineAnchorHit{0u};
  std::uint16_t nHits{0u};
  std::uint8_t nPrecisionLayers{0u};
  std::uint8_t nTriggerLayers{0u};
  std::uint8_t nPhiLayers{0u};
  std::int8_t sector{0};
  bool useBeamspot{false};
  bool needLineUpdate{false};
  bool isOverlap{false};

  __device__ std::uint32_t nBendingLayers() const {
    return nPrecisionLayers + nTriggerLayers;
  }
  __device__ double meanResidual2() const {
    return meanNormResidual2 / nBendingLayers();
  }
};

/// The result of PatternState's line test
enum class Decision : std::uint8_t { addHit, branchPattern, rejectHit };
struct LineTest {
  double residual{0.};
  double sigma{0.};
  Decision decision{Decision::rejectHit};
};

/// The quantities of a pattern that the comparisons between two patterns
/// read. The hits are read through a function from the position in the
/// pattern's hit list to the payload row
struct PatternInfo {
  std::int8_t sector{0};
  double patTheta{0.};
  double patPhi{0.};
  std::uint32_t nPhiLayers{0u};
  std::uint32_t nPrecisionLayers{0u};
  std::uint32_t nTriggerLayers{0u};
  /// PatternState::getMeanResidual2
  double meanResidual2{0.};
  std::uint32_t nHits{0u};

  __device__ std::uint32_t nBendingLayers() const {
    return nPrecisionLayers + nTriggerLayers;
  }
};

/// PatternState::countGroups: the number of groups with at least minHits hits
template <typename RowOf>
__device__ std::uint32_t countGroups(const CudaHitPayloadArrays& hits,
                                     std::uint32_t nHits, RowOf rowOf,
                                     std::uint32_t minHits) {
  std::uint32_t counts[Math::Station::nGroups]{};
  for (std::uint32_t i = 0u; i < nHits; ++i) {
    ++counts[hits.station[rowOf(i)]];
  }
  std::uint32_t nGroups{0u};
  for (std::uint32_t g = 0u; g < Math::Station::nGroups; ++g) {
    if (counts[g] > 0u && counts[g] >= minHits) {
      ++nGroups;
    }
  }
  return nGroups;
}

/// GlobalPatternFinder::isBetter
__device__ bool isBetter(const PatternInfo& a, const PatternInfo& b) {
  const double resA{a.meanResidual2};
  const double resB{b.meanResidual2};
  const double resDiff{std::abs(resA - resB) / (resA < resB ? resB : resA)};
  const int nLayerDiff{static_cast<int>(a.nBendingLayers()) -
                       static_cast<int>(b.nBendingLayers())};
  const int nPrecLayDiff{static_cast<int>(a.nPrecisionLayers) -
                         static_cast<int>(b.nPrecisionLayers)};
  if ((nLayerDiff == 0 && nPrecLayDiff == 0) ||
      (std::abs(nLayerDiff) < 3 && resDiff > 0.1)) {
    return resA < resB;
  }
  if (nLayerDiff == 0) {
    return nPrecLayDiff > 0;
  }
  return nLayerDiff > 0;
}

/// The comparison of GlobalPatternFinder::resolveOverlaps: more good groups
/// first, then isBetter
__device__ bool isBetterOverlap(const PatternInfo& a, std::uint32_t goodA,
                                const PatternInfo& b, std::uint32_t goodB) {
  if (goodA != goodB) {
    return goodA > goodB;
  }
  return isBetter(a, b);
}

/// The overlap test of GlobalPatternFinder::resolveOverlaps. goodA and goodB
/// are the numbers of good groups of the two patterns
template <typename RowOfA, typename RowOfB>
__device__ bool areOverlapping(const CudaHitPayloadArrays& hits,
                               const CudaPatternFinderConfig& cfg,
                               const PatternInfo& a, RowOfA rowOfA,
                               std::uint32_t goodA, const PatternInfo& b,
                               RowOfB rowOfB, std::uint32_t goodB) {
  namespace Sector = Math::ExpandedSector;
  if (!Sector::isNeighbour(a.sector, b.sector)) {
    return false;
  }
  if (std::abs(a.patTheta - b.patTheta) > 2. * cfg.maxThetaSizeOverlap) {
    return false;
  }
  if (a.nPhiLayers > 0u && b.nPhiLayers > 0u) {
    if (std::abs(Math::differencePeriodic(a.patPhi, b.patPhi, 2. * Math::pi)) >
        5. * Math::pi / 180.) {
      return false;
    }
  } else if (a.nPhiLayers > 0u) {
    if (!Sector::inside(b.sector, a.patPhi)) {
      return false;
    }
  } else if (b.nPhiLayers > 0u) {
    if (!Sector::inside(a.sector, b.patPhi)) {
      return false;
    }
  }
  // HitPayload::overlaps: the same measurement in two space points
  auto overlaps = [&hits](std::uint32_t row1, std::uint32_t row2) {
    return hits.geometryId[row1] == hits.geometryId[row2] &&
           std::abs(hits.localPositionY[row1] - hits.localPositionY[row2]) <
               Math::epsilon;
  };
  std::uint32_t nSharedHits{0u};
  std::uint32_t nSharedGroups{0u};
  for (std::uint32_t g = 0u; g < Math::Station::nGroups; ++g) {
    std::uint32_t nSharedInGroup{0u};
    bool inA{false};
    bool inB{false};
    for (std::uint32_t j = 0u; j < b.nHits; ++j) {
      inB = inB || hits.station[rowOfB(j)] == g;
    }
    for (std::uint32_t i = 0u; i < a.nHits && inB; ++i) {
      const std::uint32_t rowA{rowOfA(i)};
      if (hits.station[rowA] != g) {
        continue;
      }
      inA = true;
      for (std::uint32_t j = 0u; j < b.nHits; ++j) {
        const std::uint32_t rowB{rowOfB(j)};
        if (hits.station[rowB] == g && overlaps(rowA, rowB)) {
          ++nSharedInGroup;
          break;
        }
      }
    }
    if (!inA || !inB) {
      continue;
    }
    nSharedHits += nSharedInGroup;
    if (nSharedInGroup >= cfg.minGroupLayers) {
      ++nSharedGroups;
    }
  }
  const std::uint32_t minHits{a.nBendingLayers() < b.nBendingLayers()
                                  ? a.nBendingLayers()
                                  : b.nBendingLayers()};
  const std::uint32_t minGroups{goodA < goodB ? goodA : goodB};
  return nSharedHits >= 0.5 * minHits &&
         nSharedGroups >= (minGroups < 2u ? minGroups : 2u);
}

/// Everything one thread needs to build the patterns of its seed. The working
/// patterns live in four regions of `capacity` slots each: two that the
/// extension swaps between, one for the forward patterns and one for the
/// results. Each slot owns maxHits hit positions
struct SeedBuilder {
  static constexpr std::uint32_t regionA = 0u;
  static constexpr std::uint32_t regionB = 1u;
  static constexpr std::uint32_t regionForward = 2u;
  static constexpr std::uint32_t regionResult = 3u;
  static constexpr std::uint32_t nRegions = 4u;

  const CudaHitPayloadArrays& hits;
  const CudaPatternFinderConfig& cfg;
  const std::uint32_t* rows;
  const std::uint16_t* globLayers;
  Pattern* patterns;
  std::uint16_t* patternHits;
  std::uint32_t capacity;
  std::uint32_t maxHits;
  bool overflow{false};

  __device__ Pattern& pattern(std::uint32_t region, std::uint32_t k) {
    return patterns[region * capacity + k];
  }
  __device__ std::uint16_t* hitsOf(const Pattern& pat) {
    return patternHits + (&pat - patterns) * maxHits;
  }
  __device__ Math::Hit hit(std::uint16_t pos) const {
    return Math::loadHit(hits, rows[pos]);
  }
  __device__ std::uint8_t group(std::uint16_t pos) const {
    return hits.station[rows[pos]];
  }
  __device__ std::uint16_t globLayer(std::uint16_t pos) const {
    return globLayers[pos];
  }
  __device__ bool measuresPhi(std::uint16_t pos) const {
    return (hits.flags[rows[pos]] & CudaHitFlags::measuresPhi) != 0u;
  }
  __device__ bool isPrecision(std::uint16_t pos) const {
    return (hits.flags[rows[pos]] & CudaHitFlags::precision) != 0u;
  }

  /// Copies a pattern and its hits into another slot
  __device__ void copy(const Pattern& from, Pattern& to) {
    if (&from == &to) {
      return;
    }
    to = from;
    const std::uint16_t* src{hitsOf(from)};
    std::uint16_t* dst{hitsOf(to)};
    for (std::uint16_t i = 0u; i < from.nHits; ++i) {
      dst[i] = src[i];
    }
  }

  /// Appends a copy of a pattern to a region holding n patterns. Returns the
  /// copy, or nullptr if the region is full
  __device__ Pattern* push(const Pattern& pat, std::uint32_t region,
                           std::uint32_t& n) {
    if (n == capacity) {
      overflow = true;
      return nullptr;
    }
    Pattern& to{pattern(region, n++)};
    copy(pat, to);
    return &to;
  }

  /// The payload row of each hit of a pattern
  __device__ auto rowsOf(const Pattern& pat) {
    const std::uint16_t* hitPos{hitsOf(pat)};
    return [this, hitPos](std::uint32_t i) { return rows[hitPos[i]]; };
  }
  __device__ PatternInfo info(const Pattern& pat) const {
    return PatternInfo{pat.sector,           pat.patTheta,
                       pat.patPhi,           pat.nPhiLayers,
                       pat.nPrecisionLayers, pat.nTriggerLayers,
                       pat.meanResidual2(),  pat.nHits};
  }
  /// PatternState::countGroups
  __device__ std::uint32_t countGroups(const Pattern& pat, bool onlyGood) {
    return ::countGroups(hits, pat.nHits, rowsOf(pat),
                         onlyGood ? cfg.minGroupLayers : 1u);
  }

  /// PatternState::updatePatternPhi. The phi hits are summed group by group,
  /// in the order they entered each group, as on the CPU
  __device__ void updatePatternPhi(Pattern& pat) {
    namespace Sector = Math::ExpandedSector;
    if (pat.nPhiLayers == 0u) {
      pat.patPhi = Sector::phi(pat.sector);
      pat.patPhiCov = Math::square(Sector::size(pat.sector)) / 3.;
    } else {
      double sumSin{0.};
      double sumCos{0.};
      double sumWeight{0.};
      const std::uint16_t* hitPos{hitsOf(pat)};
      for (std::uint32_t g = 0u; g < Math::Station::nGroups; ++g) {
        for (std::uint16_t i = 0u; i < pat.nHits; ++i) {
          const std::uint16_t pos{hitPos[i]};
          if (group(pos) != g || !measuresPhi(pos)) {
            continue;
          }
          const Math::Hit h{hit(pos)};
          // The CPU throws on a phi hit without phi variance
          if (h.phiVariance < Math::epsilon) {
            continue;
          }
          const double w{1. / h.phiVariance};
          const double phi{Math::phi(h.position)};
          sumSin += w * std::sin(phi);
          sumCos += w * std::cos(phi);
          sumWeight += w;
        }
      }
      pat.patPhi = std::atan2(sumSin, sumCos);
      pat.patPhiCov = 1. / sumWeight;
    }
    pat.bendPlaneNorm =
        Math::directionFromPhiTheta(pat.patPhi + Math::pi / 2., Math::pi / 2.);
  }

  /// PatternState::moveLineAnchorHit: the anchor is the hit closest in layer
  /// to the reference hit, in the group closest to it other than its own
  __device__ void moveLineAnchorHit(Pattern& pat, std::uint16_t refHit) {
    if (countGroups(pat, false) < 2u) {
      pat.lineAnchorHit = pat.lastInsertedHit;
      return;
    }
    auto layerDist = [](int a, int b) { return a > b ? a - b : b - a; };
    const int refLayer{globLayer(refHit)};
    const std::uint8_t refGroup{group(refHit)};
    const std::uint16_t* hitPos{hitsOf(pat)};
    int bestDist{INT_MAX};
    std::uint32_t bestGroup{0u};
    for (std::uint32_t g = 0u; g < Math::Station::nGroups; ++g) {
      if (g == refGroup) {
        continue;
      }
      // The first hit that entered the group
      for (std::uint16_t i = 0u; i < pat.nHits; ++i) {
        if (group(hitPos[i]) == g) {
          const int dist{layerDist(globLayer(hitPos[i]), refLayer)};
          if (dist < bestDist) {
            bestDist = dist;
            bestGroup = g;
          }
          break;
        }
      }
    }
    int bestHitDist{INT_MAX};
    for (std::uint16_t i = 0u; i < pat.nHits; ++i) {
      if (group(hitPos[i]) != bestGroup) {
        continue;
      }
      const int dist{layerDist(globLayer(hitPos[i]), refLayer)};
      if (dist < bestHitDist) {
        bestHitDist = dist;
        pat.lineAnchorHit = hitPos[i];
      }
    }
  }

  /// PatternState::updateLineParameters
  __device__ void updateLineParameters(Pattern& pat) {
    if (!pat.needLineUpdate) {
      return;
    }
    const Math::Line line{Math::lineThroughHits(
        Math::projectToPhiPlane(hit(pat.lineAnchorHit), pat.bendPlaneNorm),
        Math::projectToPhiPlane(hit(pat.lastInsertedHit), pat.bendPlaneNorm),
        cfg.minHitDistance4Line, cfg.beamspot)};
    pat.linePos = line.position;
    pat.lineDir = line.direction;
    pat.leverArm = line.leverArm;
    pat.useBeamspot = line.useBeamspot;
    pat.needLineUpdate = false;
  }

  /// PatternState::isPhiCompatible
  __device__ bool isPhiCompatible(const Pattern& pat, const Math::Hit& h) {
    const double testPhi{Math::phi(h.position)};
    if (pat.nPhiLayers > 0u) {
      const double deltaPhiSigma{std::sqrt(pat.patPhiCov + h.phiVariance)};
      const double deltaPhi{
          Math::differencePeriodic(pat.patPhi, testPhi, 2. * Math::pi)};
      return !(std::abs(deltaPhi) > cfg.nPhiSigma * deltaPhiSigma);
    }
    return Math::ExpandedSector::inside(pat.sector, testPhi);
  }

  /// GlobalPatternFinder::checkLineCompatibility
  __device__ LineTest checkLine(Pattern& pat, std::uint16_t test) {
    const Math::Hit testHit{hit(test)};
    if (measuresPhi(test) && !isPhiCompatible(pat, testHit)) {
      return LineTest{};
    }
    auto makeResult = [&](Decision decision) {
      const Math::LineResidual res{Math::lineResidual(
          Math::Line{pat.linePos, pat.lineDir, pat.leverArm, pat.useBeamspot},
          pat.bendPlaneNorm, pat.patPhiCov, pat.nPhiLayers,
          hit(pat.lineAnchorHit), hit(pat.lastInsertedHit), testHit,
          cfg.beamspot)};
      LineTest out{res.residual, res.sigma, Decision::rejectHit};
      double accWindow{cfg.nResidualSigma * res.sigma};
      if (pat.useBeamspot || group(test) != group(pat.lastInsertedHit) ||
          (group(test) != group(pat.prevLayerHit) &&
           globLayer(test) == globLayer(pat.lastInsertedHit))) {
        accWindow *= 2.;
      }
      if (res.residual < accWindow) {
        out.decision = decision;
      }
      return out;
    };
    if (globLayer(test) != globLayer(pat.lastInsertedHit)) {
      updateLineParameters(pat);
      return makeResult(Decision::addHit);
    }
    if (test == pat.lastInsertedHit) {
      return LineTest{};
    }
    if (globLayer(pat.lineAnchorHit) == globLayer(pat.lastInsertedHit)) {
      return LineTest{};
    }
    return makeResult(Decision::branchPattern);
  }

  /// PatternState::addHit
  __device__ void addHit(Pattern& pat, std::uint16_t test, double residual,
                         double sigma) {
    if (pat.nHits == maxHits) {
      overflow = true;
      return;
    }
    hitsOf(pat)[pat.nHits++] = test;
    if (isPrecision(test)) {
      ++pat.nPrecisionLayers;
    } else {
      ++pat.nTriggerLayers;
    }
    if (measuresPhi(test)) {
      ++pat.nPhiLayers;
      updatePatternPhi(pat);
    }
    const bool isNewGroup{group(test) != group(pat.lastInsertedHit)};
    pat.prevLayerHit = pat.lastInsertedHit;
    pat.lastInsertedHit = test;

    pat.meanNormResidual2 += Math::square(residual / sigma);
    pat.lastResSigma = sigma;
    pat.lastResidual = residual;
    if (isNewGroup) {
      moveLineAnchorHit(pat, test);
    }
    pat.needLineUpdate = true;
  }

  /// PatternState::overWriteHit: the last inserted hit is replaced by a hit
  /// on the same layer
  __device__ void overWriteHit(Pattern& pat, std::uint16_t test,
                               double residual, double sigma) {
    const std::uint16_t last{pat.lastInsertedHit};
    if (isPrecision(last) != isPrecision(test)) {
      if (isPrecision(test)) {
        ++pat.nPrecisionLayers;
        --pat.nTriggerLayers;
      } else {
        --pat.nPrecisionLayers;
        ++pat.nTriggerLayers;
      }
    }
    bool updatePhi{false};
    if (measuresPhi(last)) {
      --pat.nPhiLayers;
      updatePhi = true;
    }
    if (measuresPhi(test)) {
      ++pat.nPhiLayers;
      updatePhi = true;
    }
    pat.meanNormResidual2 += Math::square(residual / sigma) -
                             Math::square(pat.lastResidual / pat.lastResSigma);
    pat.lastResSigma = sigma;
    pat.lastResidual = residual;

    std::uint16_t* hitPos{hitsOf(pat)};
    for (std::uint16_t i = pat.nHits; i-- > 0u;) {
      if (hitPos[i] == last) {
        hitPos[i] = test;
        break;
      }
    }
    pat.lastInsertedHit = test;
    if (updatePhi) {
      updatePatternPhi(pat);
    }
    pat.needLineUpdate = true;
  }

  /// The overlap test of GlobalPatternFinder::resolveOverlaps
  __device__ bool areOverlapping(const Pattern& a, const Pattern& b) {
    return ::areOverlapping(hits, cfg, info(a), rowsOf(a), countGroups(a, true),
                            info(b), rowsOf(b), countGroups(b, true));
  }

  /// GlobalPatternFinder::resolveOverlaps on the first n patterns of a region,
  /// in place. Returns the number of survivors, which keep their order
  __device__ std::uint32_t resolveOverlaps(std::uint32_t region,
                                           std::uint32_t n) {
    std::uint32_t nOut{0u};
    for (std::uint32_t i = 0u; i < n; ++i) {
      Pattern& pi{pattern(region, i)};
      if (pi.isOverlap) {
        continue;
      }
      for (std::uint32_t j = i + 1u; j < n; ++j) {
        Pattern& pj{pattern(region, j)};
        if (pj.isOverlap || !areOverlapping(pi, pj)) {
          continue;
        }
        if (isBetterOverlap(info(pi), countGroups(pi, true), info(pj),
                            countGroups(pj, true))) {
          pj.isOverlap = true;
        } else {
          pi.isOverlap = true;
          break;
        }
      }
      if (!pi.isOverlap) {
        copy(pi, pattern(region, nOut++));
      }
    }
    return nOut;
  }

  /// GlobalPatternFinder::extendPatterns: tests one hit against the n patterns
  /// of region `from` and writes the continued patterns to region `to`.
  /// Returns their number
  __device__ std::uint32_t extendPatterns(std::uint32_t from, std::uint32_t n,
                                          std::uint32_t to,
                                          std::uint16_t test) {
    const int testLayer{globLayer(test)};
    auto missedLayers = [&](const Pattern& pat) {
      const int d{static_cast<int>(globLayer(pat.lastInsertedHit)) - testLayer};
      return static_cast<std::uint32_t>(d < 0 ? -d : d);
    };
    std::uint32_t minMissedLayers{UINT_MAX};
    bool anyLong{false};
    for (std::uint32_t i = 0u; i < n; ++i) {
      const Pattern& pat{pattern(from, i)};
      const std::uint32_t missed{missedLayers(pat)};
      minMissedLayers = missed < minMissedLayers ? missed : minMissedLayers;
      anyLong = anyLong || pat.nBendingLayers() > 2u;
    }
    const bool shouldPrune{n > 1u && anyLong};
    const std::uint32_t maxMissed{cfg.maxMissLayersInGroup > minMissedLayers
                                      ? cfg.maxMissLayersInGroup
                                      : minMissedLayers};

    std::uint32_t nOut{0u};
    for (std::uint32_t i = 0u; i < n && !overflow; ++i) {
      Pattern& pat{pattern(from, i)};
      if (pat.isOverlap) {
        continue;
      }
      if (group(pat.lastInsertedHit) == group(test) &&
          missedLayers(pat) > maxMissed) {
        continue;
      }
      // Keep only the best of the patterns that end on the same hit
      if (shouldPrune && globLayer(pat.lastInsertedHit) != globLayer(test)) {
        bool beaten{false};
        for (std::uint32_t j = i + 1u; j < n; ++j) {
          Pattern& other{pattern(from, j)};
          if (other.lastInsertedHit != pat.lastInsertedHit || other.isOverlap) {
            continue;
          }
          if (isBetter(info(pat), info(other))) {
            other.isOverlap = true;
          } else {
            beaten = true;
            break;
          }
        }
        if (beaten) {
          continue;
        }
      }
      const LineTest res{checkLine(pat, test)};
      switch (res.decision) {
        case Decision::addHit: {
          const bool lowConfidence{res.sigma > cfg.lowConfidenceResSigma &&
                                   res.residual / res.sigma > 2.};
          if (!lowConfidence) {
            addHit(pat, test, res.residual, res.sigma);
            break;
          }
          // Keep both the pattern with the hit and the one without, unless
          // the one with the hit exists already
          bool exists{false};
          for (std::uint32_t k = 0u; k < nOut; ++k) {
            const Pattern& p{pattern(to, k)};
            exists =
                exists || (p.lastInsertedHit == test &&
                           (p.prevLayerHit == pat.lastInsertedHit ||
                            p.nBendingLayers() > pat.nBendingLayers() + 1u));
          }
          if (!exists) {
            if (Pattern * fork{push(pat, to, nOut)}; fork != nullptr) {
              addHit(*fork, test, res.residual, res.sigma);
            }
          }
          break;
        }
        case Decision::branchPattern: {
          bool exists{false};
          for (std::uint32_t k = 0u; k < nOut; ++k) {
            const Pattern& p{pattern(to, k)};
            exists = exists || (p.lastInsertedHit == test &&
                                p.prevLayerHit == pat.prevLayerHit);
          }
          if (!exists) {
            if (Pattern * branch{push(pat, to, nOut)}; branch != nullptr) {
              overWriteHit(*branch, test, res.residual, res.sigma);
            }
          }
          break;
        }
        case Decision::rejectHit:
          break;
      }
      push(pat, to, nOut);
    }
    return nOut;
  }

  /// The processHitRange helper of findPatternsInEta: extends the pattern in
  /// region A, slot 0 through the candidates first, first + step, ... up to
  /// (excluding) last, skipping the seed layer. Returns the region holding the
  /// resulting patterns, n is set to their number
  __device__ std::uint32_t processHitRange(int first, int last, int step,
                                           std::uint16_t seedPos,
                                           std::uint32_t& n) {
    std::uint32_t current{regionA};
    std::uint32_t next{regionB};
    n = 1u;
    for (int pos = first; pos != last && !overflow; pos += step) {
      const auto test = static_cast<std::uint16_t>(pos);
      if (globLayer(test) == globLayer(seedPos)) {
        continue;
      }
      n = extendPatterns(current, n, next, test);
      const std::uint32_t swap{current};
      current = next;
      next = swap;
    }
    if (n > 1u && !overflow) {
      n = resolveOverlaps(current, n);
    }
    return current;
  }

  /// The constructor of PatternState
  __device__ void makeSeedPattern(Pattern& pat, std::uint16_t seedPos,
                                  std::int8_t sector) {
    pat = Pattern{};
    pat.lastInsertedHit = seedPos;
    pat.prevLayerHit = seedPos;
    pat.lineAnchorHit = seedPos;
    pat.patTheta = Math::theta(hit(seedPos).position);
    pat.sector = sector;
    hitsOf(pat)[0] = seedPos;
    pat.nHits = 1u;
    if (isPrecision(seedPos)) {
      ++pat.nPrecisionLayers;
    } else {
      ++pat.nTriggerLayers;
    }
    if (measuresPhi(seedPos)) {
      ++pat.nPhiLayers;
    }
    updatePatternPhi(pat);
    pat.needLineUpdate = true;
  }

  /// GlobalPatternFinder::passPatternCuts
  __device__ bool passPatternCuts(const Pattern& pat) {
    return pat.nTriggerLayers >= cfg.minStripEtaLayers &&
           pat.nPrecisionLayers >= cfg.minPrecisionLayers &&
           countGroups(pat, true) >= 2u &&
           !(pat.meanNormResidual2 > cfg.meanNormRes2Cut);
  }

  /// Builds the patterns of a seed into the result region. Returns their
  /// number
  __device__ std::uint32_t build(std::uint16_t seedPos, std::uint32_t nCands,
                                 std::int8_t sector) {
    makeSeedPattern(pattern(regionA, 0u), seedPos, sector);

    // First from the seed layer outwards
    std::uint32_t nForward{0u};
    const std::uint32_t forward{processHitRange(
        seedPos + 1, static_cast<int>(nCands), 1, seedPos, nForward)};
    for (std::uint32_t k = 0u; k < nForward; ++k) {
      copy(pattern(forward, k), pattern(regionForward, k));
    }

    // Then from the seed inwards, once per forward pattern
    std::uint32_t nResult{0u};
    for (std::uint32_t f = 0u; f < nForward && !overflow; ++f) {
      Pattern& start{pattern(regionA, 0u)};
      copy(pattern(regionForward, f), start);
      moveLineAnchorHit(start, seedPos);
      start.lastInsertedHit = seedPos;
      std::uint32_t nBackward{0u};
      const std::uint32_t backward{
          processHitRange(seedPos - 1, -1, -1, seedPos, nBackward)};
      for (std::uint32_t k = 0u; k < nBackward && !overflow; ++k) {
        push(pattern(backward, k), regionResult, nResult);
      }
    }
    if (nResult > 1u && !overflow) {
      nResult = resolveOverlaps(regionResult, nResult);
    }
    return overflow ? 0u : nResult;
  }
};

/// Writes a finalized pattern to an output slot. The hit references become
/// positions in the pattern's own hit list
__device__ void writePattern(const SeedBuilder& builder, const Pattern& pat,
                             const std::uint16_t* hitPos,
                             const CudaPatternRecordArrays& out,
                             std::uint32_t slot) {
  auto ownPosition = [&](std::uint16_t candidate) -> std::uint16_t {
    for (std::uint16_t i = 0u; i < pat.nHits; ++i) {
      if (hitPos[i] == candidate) {
        return i;
      }
    }
    return cudaInvalidHitPosition;
  };
  out.sector[slot] = pat.sector;
  out.patPhi[slot] = pat.patPhi;
  out.patTheta[slot] = pat.patTheta;
  out.patPhiCov[slot] = pat.patPhiCov;
  out.meanNormResidual2[slot] = pat.meanNormResidual2;
  out.lastResidual[slot] = pat.lastResidual;
  out.lastResSigma[slot] = pat.lastResSigma;
  out.linePosX[slot] = pat.linePos.x;
  out.linePosY[slot] = pat.linePos.y;
  out.linePosZ[slot] = pat.linePos.z;
  out.lineDirX[slot] = pat.lineDir.x;
  out.lineDirY[slot] = pat.lineDir.y;
  out.lineDirZ[slot] = pat.lineDir.z;
  out.leverArm[slot] = pat.leverArm;
  out.bendPlaneNormX[slot] = pat.bendPlaneNorm.x;
  out.bendPlaneNormY[slot] = pat.bendPlaneNorm.y;
  out.bendPlaneNormZ[slot] = pat.bendPlaneNorm.z;
  out.nPrecisionLayers[slot] = pat.nPrecisionLayers;
  out.nTriggerLayers[slot] = pat.nTriggerLayers;
  out.nPhiLayers[slot] = pat.nPhiLayers;
  out.nHits[slot] = pat.nHits;
  out.lastInsertedHit[slot] = ownPosition(pat.lastInsertedHit);
  out.prevLayerHit[slot] = ownPosition(pat.prevLayerHit);
  out.lineAnchorHit[slot] = ownPosition(pat.lineAnchorHit);
  out.flags[slot] =
      CudaPatternFlags::isFinalized |
      (pat.useBeamspot ? CudaPatternFlags::useBeamspot : 0u) |
      (pat.needLineUpdate ? CudaPatternFlags::needLineUpdate : 0u);
  out.status[slot] = CudaPatternStatus::ok;
  for (std::uint16_t i = 0u; i < pat.nHits; ++i) {
    out.hitIndex[out.hitSlot(slot, i)] = builder.rows[hitPos[i]];
    out.globLayer[out.hitSlot(slot, i)] = builder.globLayers[hitPos[i]];
  }
}

__global__ void buildPatternsKernel(
    CudaHitPayloadArrays hits, CudaCandidateArrays lists,
    CudaPatternFinderConfig cfg, std::uint32_t firstSeed,
    CudaPatternRecordArrays out, const std::uint8_t* onlyIf,
    Pattern* workPatterns, std::uint16_t* workHits) {
  // Entry b of the records is seed firstSeed + b
  const std::uint32_t b{blockIdx.x * blockDim.x + threadIdx.x};
  const std::uint32_t seed{firstSeed + b};
  if (b >= out.nSeeds || seed >= lists.nSeeds ||
      (onlyIf != nullptr && onlyIf[b] == 0u)) {
    return;
  }
  out.nPatterns[b] = 0u;
  out.overflow[b] = 0u;
  const std::uint16_t seedPos{lists.seedPosition[seed]};
  if (lists.seedStatus[seed] != CudaSeedStatus::accepted ||
      seedPos == cudaInvalidHitPosition) {
    return;
  }
  const std::uint32_t nWork{SeedBuilder::nRegions * out.capacityPerSeed};
  SeedBuilder builder{
      .hits = hits,
      .cfg = cfg,
      .rows = lists.candidateIndices + lists.begin(seed),
      .globLayers = lists.candidateGlobLayer + lists.begin(seed),
      .patterns = workPatterns + b * nWork,
      .patternHits = workHits + b * nWork * out.maxHitsPerPattern,
      .capacity = out.capacityPerSeed,
      .maxHits = out.maxHitsPerPattern};
  const std::uint32_t nResult{
      builder.build(seedPos, lists.end(seed) - lists.begin(seed),
                    static_cast<std::int8_t>(lists.seedSector[seed]))};
  if (builder.overflow) {
    out.overflow[b] = 1u;
    return;
  }

  std::uint32_t nOut{0u};
  for (std::uint32_t k = 0u; k < nResult; ++k) {
    Pattern& pat{builder.pattern(SeedBuilder::regionResult, k)};
    pat.meanNormResidual2 /= pat.nBendingLayers();
    if (!builder.passPatternCuts(pat)) {
      continue;
    }
    writePattern(builder, pat, builder.hitsOf(pat), out, out.slot(b, nOut));
    ++nOut;
  }
  out.nPatterns[b] = nOut;
}

/// Copies a pattern record between two containers
__device__ void copyRecord(const CudaPatternRecordArrays& from,
                           std::uint32_t fromSlot,
                           const CudaPatternRecordArrays& to,
                           std::uint32_t toSlot) {
  to.sector[toSlot] = from.sector[fromSlot];
  to.patPhi[toSlot] = from.patPhi[fromSlot];
  to.patTheta[toSlot] = from.patTheta[fromSlot];
  to.patPhiCov[toSlot] = from.patPhiCov[fromSlot];
  to.meanNormResidual2[toSlot] = from.meanNormResidual2[fromSlot];
  to.lastResidual[toSlot] = from.lastResidual[fromSlot];
  to.lastResSigma[toSlot] = from.lastResSigma[fromSlot];
  to.linePosX[toSlot] = from.linePosX[fromSlot];
  to.linePosY[toSlot] = from.linePosY[fromSlot];
  to.linePosZ[toSlot] = from.linePosZ[fromSlot];
  to.lineDirX[toSlot] = from.lineDirX[fromSlot];
  to.lineDirY[toSlot] = from.lineDirY[fromSlot];
  to.lineDirZ[toSlot] = from.lineDirZ[fromSlot];
  to.leverArm[toSlot] = from.leverArm[fromSlot];
  to.bendPlaneNormX[toSlot] = from.bendPlaneNormX[fromSlot];
  to.bendPlaneNormY[toSlot] = from.bendPlaneNormY[fromSlot];
  to.bendPlaneNormZ[toSlot] = from.bendPlaneNormZ[fromSlot];
  to.nPrecisionLayers[toSlot] = from.nPrecisionLayers[fromSlot];
  to.nTriggerLayers[toSlot] = from.nTriggerLayers[fromSlot];
  to.nPhiLayers[toSlot] = from.nPhiLayers[fromSlot];
  to.nHits[toSlot] = from.nHits[fromSlot];
  to.lastInsertedHit[toSlot] = from.lastInsertedHit[fromSlot];
  to.prevLayerHit[toSlot] = from.prevLayerHit[fromSlot];
  to.lineAnchorHit[toSlot] = from.lineAnchorHit[fromSlot];
  to.flags[toSlot] = from.flags[fromSlot];
  to.status[toSlot] = from.status[fromSlot];
  for (std::uint32_t i = 0u; i < from.nHits[fromSlot]; ++i) {
    to.hitIndex[to.hitSlot(toSlot, i)] =
        from.hitIndex[from.hitSlot(fromSlot, i)];
    to.globLayer[to.hitSlot(toSlot, i)] =
        from.globLayer[from.hitSlot(fromSlot, i)];
  }
}

/// A finalized pattern record seen by the comparisons
__device__ PatternInfo recordInfo(const CudaPatternRecordArrays& r,
                                  std::uint32_t slot) {
  return PatternInfo{r.sector[slot],
                     r.patTheta[slot],
                     r.patPhi[slot],
                     r.nPhiLayers[slot],
                     r.nPrecisionLayers[slot],
                     r.nTriggerLayers[slot],
                     r.meanNormResidual2[slot],
                     r.nHits[slot]};
}

/// Takes the patterns of one batch of seeds into the event's patterns, one
/// seed after the other in tree order. A seed whose hit is already in
/// maxSeedAttempts accepted patterns is skipped. A new pattern that overlaps
/// a better accepted pattern is dropped; otherwise it is accepted and the
/// accepted patterns it overlaps are dropped. The accepted patterns therefore
/// never overlap, and no overlap removal over all of them is needed. Dropped
/// patterns keep their slot with status rejected. Runs in one thread.
///
/// The patterns of seed b of the batch are in `first`, or in `retry` if the
/// first build overflowed. If the retry overflowed too, or the event's
/// patterns are full, the overflow flag of the event is raised.
__global__ void acceptPatternsKernel(CudaHitPayloadArrays hits,
                                     CudaCandidateArrays lists,
                                     CudaPatternFinderConfig cfg,
                                     std::uint32_t firstSeed,
                                     CudaPatternRecordArrays first,
                                     CudaPatternRecordArrays retry,
                                     CudaPatternRecordArrays event) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) {
    return;
  }
  auto rowsOf = [](const CudaPatternRecordArrays& r, std::uint32_t slot) {
    return
        [&r, slot](std::uint32_t i) { return r.hitIndex[r.hitSlot(slot, i)]; };
  };
  auto isAlive = [&](std::uint32_t slot) {
    return event.status[slot] == CudaPatternStatus::ok;
  };
  std::uint32_t nEvent{event.nPatterns[0]};
  for (std::uint32_t b = 0u; b < first.nSeeds; ++b) {
    const std::uint32_t seed{firstSeed + b};
    if (seed >= lists.nSeeds ||
        lists.seedStatus[seed] != CudaSeedStatus::accepted) {
      continue;
    }
    const CudaPatternRecordArrays& from{first.overflow[b] != 0u ? retry
                                                                : first};
    if (from.overflow[b] != 0u) {
      event.overflow[0] = 1u;
      continue;
    }
    // The skip rule, against the accepted patterns
    const std::uint32_t seedRow{lists.seedHitIndex[seed]};
    std::uint32_t nContaining{0u};
    for (std::uint32_t e = 0u; e < nEvent; ++e) {
      if (!isAlive(e)) {
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

    for (std::uint32_t k = 0u; k < from.nPatterns[b]; ++k) {
      const std::uint32_t slot{from.slot(b, k)};
      const PatternInfo info{recordInfo(from, slot)};
      const std::uint32_t good{countGroups(hits, info.nHits, rowsOf(from, slot),
                                           cfg.minGroupLayers)};
      auto overlapsWith = [&](std::uint32_t e, std::uint32_t& goodE) {
        goodE = countGroups(hits, event.nHits[e], rowsOf(event, e),
                            cfg.minGroupLayers);
        return areOverlapping(hits, cfg, info, rowsOf(from, slot), good,
                              recordInfo(event, e), rowsOf(event, e), goodE);
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
        event.overflow[0] = 1u;
        break;
      }
      for (std::uint32_t e = 0u; e < nEvent; ++e) {
        std::uint32_t goodE{0u};
        if (isAlive(e) && overlapsWith(e, goodE)) {
          event.status[e] = CudaPatternStatus::rejected;
          event.flags[e] |= CudaPatternFlags::isOverlap;
        }
      }
      copyRecord(from, slot, event, nEvent++);
    }
  }
  event.nPatterns[0] = nEvent;
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
                       cudaStream_t stream,
                       const CudaPatternRecordContainer* rebuildOverflowOf) {
  if (!hits.isOnDevice() || !lists.isOnDevice() || !patterns.isOnDevice() ||
      (rebuildOverflowOf != nullptr && !rebuildOverflowOf->isOnDevice())) {
    throw std::invalid_argument(
        "buildCudaPatterns: the hits, the candidate lists and the pattern "
        "records must be on the device");
  }
  if (rebuildOverflowOf != nullptr &&
      rebuildOverflowOf->nSeeds() != patterns.nSeeds()) {
    throw std::invalid_argument(
        "buildCudaPatterns: the records to rebuild cover other seeds");
  }
  if (patterns.empty() || firstSeed >= lists.nSeeds()) {
    return;
  }
  // The working patterns of every seed, released once the kernel is done
  const std::size_t nWork{patterns.nSeeds() * SeedBuilder::nRegions *
                          patterns.capacityPerSeed()};
  Pattern* workPatterns{nullptr};
  std::uint16_t* workHits{nullptr};
  ACTS_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&workPatterns),
                                  nWork * sizeof(Pattern), stream));
  ACTS_CUDA_CHECK(cudaMallocAsync(
      reinterpret_cast<void**>(&workHits),
      nWork * patterns.maxHitsPerPattern() * sizeof(std::uint16_t), stream));

  constexpr unsigned blockSize{64u};
  const auto nBlocks{
      static_cast<unsigned>((patterns.nSeeds() + blockSize - 1u) / blockSize)};
  buildPatternsKernel<<<nBlocks, blockSize, 0, stream>>>(
      hits.deviceArrays(), lists.deviceArrays(), cfg,
      static_cast<std::uint32_t>(firstSeed), patterns.deviceArrays(),
      rebuildOverflowOf != nullptr ? rebuildOverflowOf->deviceArrays().overflow
                                   : nullptr,
      workPatterns, workHits);
  ACTS_CUDA_CHECK(cudaGetLastError());

  ACTS_CUDA_CHECK(cudaFreeAsync(workPatterns, stream));
  ACTS_CUDA_CHECK(cudaFreeAsync(workHits, stream));
}

CudaPatternRecordContainer findCudaPatternsInEta(
    const CudaHitPayloadContainer& hits, CudaCandidateListContainer& lists,
    const CudaPatternFinderConfig& cfg, CudaPatternFinderSizes sizes,
    cudaStream_t stream, CudaPatternFinderTiming* timing) {
  if (sizes.batchSize == 0u) {
    throw std::invalid_argument("findCudaPatternsInEta: empty batches");
  }
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
  auto elapsedUs = [&](std::size_t from, std::size_t to) {
    float ms{0.f};
    ACTS_CUDA_CHECK(cudaEventElapsedTime(&ms, marks[from], marks[to]));
    return 1000. * ms;
  };
  auto finish = [&](CudaPatternRecordContainer&& event, std::size_t attempt) {
    if (timing != nullptr) {
      timing->nAttempts = attempt + 1u;
      timing->nPatternsInEta = 0u;
      for (std::uint32_t k = 0u; k < event.nPatterns(0u); ++k) {
        timing->nPatternsInEta +=
            event.pattern(event.slot(0u, k)).status == CudaPatternStatus::ok;
      }
      for (cudaEvent_t& m : marks) {
        ACTS_CUDA_CHECK(cudaEventDestroy(m));
      }
    }
    return std::move(event);
  };

  if (timing != nullptr) {
    timing->kernelsUs = 0.;
    timing->downloadUs = 0.;
  }
  for (std::size_t attempt = 0u;; ++attempt) {
    mark(0u);
    prepareCudaSeeds(hits, lists, cfg, stream);
    const std::size_t batchSize{
        std::max<std::size_t>(1u, std::min(sizes.batchSize, lists.nSeeds()))};
    CudaPatternRecordContainer first{batchSize, sizes.capacityPerSeed,
                                     sizes.maxHitsPerPattern};
    CudaPatternRecordContainer retry{batchSize, sizes.retryCapacityPerSeed,
                                     sizes.maxHitsPerPattern};
    CudaPatternRecordContainer event{1u, sizes.maxPatterns,
                                     sizes.maxHitsPerPattern};
    first.moveToDevice(stream);
    retry.moveToDevice(stream);
    event.moveToDevice(stream);

    for (std::size_t firstSeed = 0u; firstSeed < lists.nSeeds();
         firstSeed += batchSize) {
      buildCudaPatterns(hits, lists, cfg, firstSeed, first, stream);
      buildCudaPatterns(hits, lists, cfg, firstSeed, retry, stream, &first);
      acceptPatternsKernel<<<1, 1, 0, stream>>>(
          hits.deviceArrays(), lists.deviceArrays(), cfg,
          static_cast<std::uint32_t>(firstSeed), first.deviceArrays(),
          retry.deviceArrays(), event.deviceArrays());
      ACTS_CUDA_CHECK(cudaGetLastError());
    }
    mark(1u);
    event.moveToHost(stream);
    mark(2u);
    ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (timing != nullptr) {
      timing->kernelsUs += elapsedUs(0u, 1u);
      timing->downloadUs += elapsedUs(1u, 2u);
    }
    if (!event.overflow(0u)) {
      return finish(std::move(event), attempt);
    }
    if (attempt + 1u >= sizes.maxAttempts) {
      for (cudaEvent_t& m : marks) {
        if (timing != nullptr) {
          ACTS_CUDA_CHECK(cudaEventDestroy(m));
        }
      }
      throw std::runtime_error(
          "findCudaPatternsInEta: the event still overflows the pattern "
          "records after the last attempt");
    }
    sizes.retryCapacityPerSeed *= 2u;
    sizes.maxHitsPerPattern = std::min<std::size_t>(
        2u * sizes.maxHitsPerPattern, cudaInvalidHitPosition - 1u);
    sizes.maxPatterns *= 2u;
  }
}

}  // namespace ActsExamples
