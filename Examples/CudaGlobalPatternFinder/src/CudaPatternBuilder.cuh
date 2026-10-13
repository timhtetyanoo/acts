// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

/// The device side of the pattern building that the one-thread-per-seed
/// finding (CudaPatternFinder.cu) and the layer-by-layer finding
/// (CudaPatternFinderV2.cu) share: the pattern being built, the operations on
/// it that GlobalPatternFinder and PatternStateAux make, and the comparisons
/// between two patterns. Included by the .cu files only.

#include "ActsExamples/TrackFinding/CudaPatternFinder.hpp"

#include <climits>
#include <cstdint>

namespace {

using namespace ActsExamples;
namespace Math = CudaPatternMath;

/// One pattern while it is built: PatternStateAux with its hits, the last
/// inserted hit, the last hit of the previous layer and the line anchor held
/// as positions in the seed's sorted candidate list. A position names both
/// the hit and its global layer, as the CPU's OrderedHit does
struct Pattern {
  Math::Vector3 linePos{};
  Math::Vector3 lineDir{};
  /// The bending plane: patPhi, patPhiOffset, bendPlaneNorm and bendPlaneCov
  Math::BendPlane plane{};
  double leverArm{0.};
  double meanNormResidual2{0.};
  double lastResidual{0.};
  double lastResSigma{0.};
  double patTheta{0.};
  std::uint32_t nDoF{0u};
  std::uint8_t lastNDoF{0u};
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
  /// The forward pattern a backward pattern descends from, in the
  /// layer-by-layer finding; it fits the padding at the end of the struct
  std::uint16_t family{0u};

  __device__ std::uint32_t nBendingLayers() const {
    return nPrecisionLayers + nTriggerLayers;
  }
  /// PatternStateAux::getMeanResidual2 of a pattern not finalized yet
  __device__ double meanResidual2() const { return meanNormResidual2 / nDoF; }
};

/// The result of the line test of GlobalPatternFinder
enum class Decision : std::uint8_t { addHit, branchPattern, rejectHit };
struct LineTest {
  double residual{0.};
  double sigma{0.};
  std::uint8_t nDof{0u};
  Decision decision{Decision::rejectHit};
};

/// The quantities of a pattern that the comparisons between two patterns
/// read. The hits are read through a function from the position in the
/// pattern's hit list to the payload row
struct PatternInfo {
  std::int8_t sector{0};
  double patTheta{0.};
  double patPhi{0.};
  double patPhiOffset{0.};
  /// Transverse distance of the line anchor hit from the beam line
  double anchorRadius{0.};
  std::uint32_t nPhiLayers{0u};
  std::uint32_t nPrecisionLayers{0u};
  std::uint32_t nTriggerLayers{0u};
  /// PatternStateAux::getMeanResidual2
  double meanResidual2{0.};
  std::uint32_t nHits{0u};

  __device__ std::uint32_t nBendingLayers() const {
    return nPrecisionLayers + nTriggerLayers;
  }
};

/// PatternStateAux::countGroups: the number of groups with at least minHits
/// hits
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
      (std::abs(nLayerDiff) < 3 && resDiff > 0.1 &&
       (resA < resB ? resA : resB) > 1.)) {
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
  // The phi of both patterns at a common radius, where the first one lives
  const double refR{a.anchorRadius};
  const double phiA{a.patPhi + a.patPhiOffset / refR};
  const double phiB{b.patPhi + b.patPhiOffset / refR};
  if (a.nPhiLayers > 1u && b.nPhiLayers > 1u) {
    if (std::abs(Math::differencePeriodic(phiA, phiB, 2. * Math::pi)) >
        2. * cfg.phiTolerance) {
      return false;
    }
  } else if (a.nPhiLayers > 0u) {
    if (!Sector::inside(b.sector, phiA)) {
      return false;
    }
  } else if (b.nPhiLayers > 0u) {
    if (!Sector::inside(a.sector, phiB)) {
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
  /// CudaPatternOverflow bits of what did not fit
  std::uint8_t overflow{0u};

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
      overflow |= CudaPatternOverflow::patterns;
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
    return PatternInfo{
        .sector = pat.sector,
        .patTheta = pat.patTheta,
        .patPhi = pat.plane.phi,
        .patPhiOffset = pat.plane.offset,
        .anchorRadius = Math::perp(hit(pat.lineAnchorHit).position),
        .nPhiLayers = pat.nPhiLayers,
        .nPrecisionLayers = pat.nPrecisionLayers,
        .nTriggerLayers = pat.nTriggerLayers,
        .meanResidual2 = pat.meanResidual2(),
        .nHits = pat.nHits};
  }
  /// PatternStateAux::countGroups
  __device__ std::uint32_t countGroups(const Pattern& pat, bool onlyGood) {
    return ::countGroups(hits, pat.nHits, rowsOf(pat),
                         onlyGood ? cfg.minGroupLayers : 1u);
  }

  /// PatternStateAux::updatePatternPhi. The phi hits are visited group by
  /// group, in the order they entered each group, as on the CPU
  __device__ void updatePatternPhi(Pattern& pat) {
    const std::uint16_t* hitPos{hitsOf(pat)};
    auto forEachPhiHit = [&](auto&& fn) {
      for (std::uint32_t g = 0u; g < Math::Station::nGroups; ++g) {
        for (std::uint16_t i = 0u; i < pat.nHits; ++i) {
          const std::uint16_t pos{hitPos[i]};
          if (group(pos) == g && measuresPhi(pos)) {
            const Math::Hit h{hit(pos)};
            fn(h);
          }
        }
      }
    };
    pat.plane = Math::fitBendPlane(pat.nPhiLayers, pat.plane.phi, pat.sector,
                                   cfg.beamspot, forEachPhiHit);
  }

  /// PatternStateAux::moveLineAnchorHit: the anchor is the hit closest in layer
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

  /// PatternStateAux::updateLineParameters
  __device__ void updateLineParameters(Pattern& pat) {
    if (!pat.needLineUpdate) {
      return;
    }
    const Math::Line line{Math::lineThroughHits(
        Math::projectToPhiPlane(hit(pat.lineAnchorHit), pat.plane.normal,
                                pat.plane.offset),
        Math::projectToPhiPlane(hit(pat.lastInsertedHit), pat.plane.normal,
                                pat.plane.offset),
        cfg.minHitDistance4Line, cfg.beamspot)};
    pat.linePos = line.position;
    pat.lineDir = line.direction;
    pat.leverArm = line.leverArm;
    pat.useBeamspot = line.useBeamspot;
    pat.needLineUpdate = false;
  }

  /// GlobalPatternFinder::isPhiCompatible
  __device__ bool isPhiCompatible(const Pattern& pat, const Math::Hit& h) {
    if (pat.nPhiLayers > 0u) {
      return !(std::abs(Math::angleToBendPlane(pat.plane, h.position)) >
               cfg.phiTolerance);
    }
    return Math::ExpandedSector::inside(pat.sector, Math::phi(h.position));
  }

  /// GlobalPatternFinder::checkLineCompatibility
  __device__ LineTest checkLine(Pattern& pat, std::uint16_t test) {
    return checkLine(pat, test, hit(test));
  }
  /// The same, with the test hit already loaded
  __device__ LineTest checkLine(Pattern& pat, std::uint16_t test,
                                const Math::Hit& testHit) {
    if ((testHit.flags & CudaHitFlags::measuresPhi) != 0u &&
        !isPhiCompatible(pat, testHit)) {
      return LineTest{};
    }
    auto makeResult = [&](Decision decision) {
      const Math::LineResidual res{Math::lineResidual(
          Math::Line{pat.linePos, pat.lineDir, pat.leverArm, pat.useBeamspot},
          pat.plane, pat.nPhiLayers, hit(pat.lineAnchorHit),
          hit(pat.lastInsertedHit), testHit, cfg.beamspot)};
      LineTest out{res.residual, res.sigma, res.nDof, Decision::rejectHit};
      double accWindow{(res.nDof == 1u ? cfg.nResidualSigma : cfg.cut2Dof) *
                       res.sigma};
      if (pat.useBeamspot || testHit.station != group(pat.lastInsertedHit) ||
          (testHit.station != group(pat.prevLayerHit) &&
           globLayer(test) == globLayer(pat.lastInsertedHit))) {
        accWindow *= cfg.loosenAcceptanceFactor;
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

  /// PatternStateAux::addHit
  __device__ void addHit(Pattern& pat, std::uint16_t test,
                         const LineTest& res) {
    if (pat.nHits == maxHits) {
      overflow |= CudaPatternOverflow::hits;
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

    pat.meanNormResidual2 += Math::square(res.residual / res.sigma);
    pat.nDoF += res.nDof;
    pat.lastResSigma = res.sigma;
    pat.lastResidual = res.residual;
    pat.lastNDoF = res.nDof;
    if (isNewGroup) {
      moveLineAnchorHit(pat, test);
    }
    pat.needLineUpdate = true;
  }

  /// PatternStateAux::overWriteHit: the last inserted hit is replaced by a
  /// hit on the same layer
  __device__ void overWriteHit(Pattern& pat, std::uint16_t test,
                               const LineTest& res) {
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
    pat.meanNormResidual2 += Math::square(res.residual / res.sigma) -
                             Math::square(pat.lastResidual / pat.lastResSigma);
    pat.nDoF += static_cast<std::uint32_t>(static_cast<int>(res.nDof) -
                                           static_cast<int>(pat.lastNDoF));
    pat.lastResSigma = res.sigma;
    pat.lastResidual = res.residual;
    pat.lastNDoF = res.nDof;

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
            addHit(pat, test, res);
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
              addHit(*fork, test, res);
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
              overWriteHit(*branch, test, res);
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

  /// The constructor of PatternStateAux
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
  out.patPhi[slot] = pat.plane.phi;
  out.patTheta[slot] = pat.patTheta;
  out.patPhiOffset[slot] = pat.plane.offset;
  out.bendPlaneCovPhiPhi[slot] = pat.plane.covPhiPhi;
  out.bendPlaneCovSS[slot] = pat.plane.covSS;
  out.bendPlaneCovPhiS[slot] = pat.plane.covPhiS;
  out.nDoF[slot] = pat.nDoF;
  out.lastNDoF[slot] = pat.lastNDoF;
  out.anchorRadius[slot] = Math::perp(builder.hit(pat.lineAnchorHit).position);
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
  out.bendPlaneNormX[slot] = pat.plane.normal.x;
  out.bendPlaneNormY[slot] = pat.plane.normal.y;
  out.bendPlaneNormZ[slot] = pat.plane.normal.z;
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

/// Copies a pattern record between two containers
__device__ void copyRecord(const CudaPatternRecordArrays& from,
                           std::uint32_t fromSlot,
                           const CudaPatternRecordArrays& to,
                           std::uint32_t toSlot) {
  to.sector[toSlot] = from.sector[fromSlot];
  to.patPhi[toSlot] = from.patPhi[fromSlot];
  to.patTheta[toSlot] = from.patTheta[fromSlot];
  to.patPhiOffset[toSlot] = from.patPhiOffset[fromSlot];
  to.bendPlaneCovPhiPhi[toSlot] = from.bendPlaneCovPhiPhi[fromSlot];
  to.bendPlaneCovSS[toSlot] = from.bendPlaneCovSS[fromSlot];
  to.bendPlaneCovPhiS[toSlot] = from.bendPlaneCovPhiS[fromSlot];
  to.nDoF[toSlot] = from.nDoF[fromSlot];
  to.lastNDoF[toSlot] = from.lastNDoF[fromSlot];
  to.anchorRadius[toSlot] = from.anchorRadius[fromSlot];
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
  return PatternInfo{.sector = r.sector[slot],
                     .patTheta = r.patTheta[slot],
                     .patPhi = r.patPhi[slot],
                     .patPhiOffset = r.patPhiOffset[slot],
                     .anchorRadius = r.anchorRadius[slot],
                     .nPhiLayers = r.nPhiLayers[slot],
                     .nPrecisionLayers = r.nPrecisionLayers[slot],
                     .nTriggerLayers = r.nTriggerLayers[slot],
                     .meanResidual2 = r.meanNormResidual2[slot],
                     .nHits = r.nHits[slot]};
}

}  // namespace
