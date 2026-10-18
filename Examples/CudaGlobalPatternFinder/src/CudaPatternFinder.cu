// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

/// @file
/// @brief Pattern finding in eta on the device: one warp per seed, one layer
/// at a time, then overlap resolution over the event by priority.

#include "ActsExamples/TrackFinding/CudaPatternFinder.hpp"
#include "ActsExamples/Utilities/CudaUtilities.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <cub/cub.cuh>

namespace {

using namespace ActsExamples;
namespace Math = CudaPatternMath;

using Real = Math::Real;
using RealHit = Math::HitT<Real>;
using RealVector3 = Math::Vector3T<Real>;

/// Bit layout of the per-candidate byte written by the seed checks
struct CandidateInfo {
  static constexpr std::uint8_t groupMask = 0x7u;
  static constexpr std::uint8_t measuresPhi = 0x8u;
  static constexpr std::uint8_t precision = 0x10u;
};
static_assert(Math::Station::nGroups <= CandidateInfo::groupMask + 1u);

/// A pattern under construction. Hits are positions in the seed's sorted
/// candidate list; the slot holds the pattern followed by its hit positions.
struct Pattern {
  RealVector3 linePos{};
  RealVector3 lineDir{};
  Math::BendPlaneT<Real> plane{};
  Real leverArm{0};
  Real sumNormResidual2{0};
  Real lastResidual{0};
  Real lastResSigma{0};
  Real patTheta{0};
  std::uint32_t nDoF{0u};
  std::uint8_t lastNDoF{0u};
  std::uint16_t lastInsertedHit{0u};
  std::uint16_t prevLayerHit{0u};
  std::uint16_t lineAnchorHit{0u};
  /// Second point of the pattern line. Stays put when a hit is added until
  /// the line is refitted
  std::uint16_t lineEndHit{0u};
  std::uint16_t nHits{0u};
  std::uint8_t nPrecisionLayers{0u};
  std::uint8_t nTriggerLayers{0u};
  std::uint8_t nPhiLayers{0u};
  /// Pending PatternState updates: bending plane, line anchor, line
  std::uint8_t needUpdate{0u};
  std::int8_t sector{0};
  bool useBeamspot{false};
  bool isOverlap{false};
  static constexpr std::uint8_t kBend = 1u << 0u;
  static constexpr std::uint8_t kAnchor = 1u << 1u;
  static constexpr std::uint8_t kLine = 1u << 2u;
  /// Forward pattern this backward pattern descends from; patterns are only
  /// pruned and compared within a family
  std::uint16_t family{0u};

  __device__ std::uint32_t nBendingLayers() const {
    return nPrecisionLayers + nTriggerLayers;
  }
  __device__ Real meanResidual2() const {
    return sumNormResidual2 / (nDoF > 1u ? static_cast<Real>(nDoF) : Real{1});
  }
};

static_assert(alignof(Pattern) <= 8u,
              "the slots of patterns are copied in 8-byte words");

enum class Decision : std::uint8_t { eAddHit, eBranchPattern, eRejectHit };
struct LineTest {
  Real residual{0};
  Real sigma{0};
  std::uint8_t nDof{0u};
  Decision decision{Decision::eRejectHit};
};

struct PatternInfo {
  std::int8_t sector{0};
  Real patTheta{0};
  Real patPhi{0};
  Real patPhiOffset{0};
  /// Transverse distance of the line anchor hit from the beam line
  Real anchorRadius{0};
  std::uint32_t nPhiLayers{0u};
  std::uint32_t nPrecisionLayers{0u};
  std::uint32_t nTriggerLayers{0u};
  Real meanResidual2{0};
  std::uint32_t nHits{0u};

  __device__ std::uint32_t nBendingLayers() const {
    return nPrecisionLayers + nTriggerLayers;
  }
};

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

/// Overlap: neighbouring sectors, close in theta and phi, and enough shared
/// hits and groups
template <typename RowOfA, typename RowOfB>
__device__ bool areOverlapping(const CudaHitPayloadArrays& hits,
                               const CudaPatternFinderConfig& cfg,
                               const PatternInfo& a, RowOfA rowOfA,
                               const PatternInfo& b, RowOfB rowOfB) {
  namespace Sector = Math::ExpandedSector;
  if (!Sector::isNeighbour(a.sector, b.sector)) {
    return false;
  }
  if (std::abs(a.patTheta - b.patTheta) >
      static_cast<Real>(cfg.maxThetaSizeOverlap)) {
    return false;
  }
  const Real refR{a.anchorRadius};
  const Real phiA{a.patPhi + a.patPhiOffset / refR};
  const Real phiB{b.patPhi + b.patPhiOffset / refR};
  if (a.nPhiLayers > 1u && b.nPhiLayers > 1u) {
    if (std::abs(Math::differencePeriodic(phiA, phiB,
                                          static_cast<Real>(2. * Math::pi))) >
        static_cast<Real>(2. * cfg.phiTolerance)) {
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
  // The same space point, or two space points from the same measurement
  auto overlaps = [&hits](std::uint32_t row1, std::uint32_t row2) {
    return row1 == row2 ||
           (hits.geometryId[row1] == hits.geometryId[row2] &&
            std::abs(hits.localPositionY[row1] - hits.localPositionY[row2]) <
                Math::epsilon);
  };
  // Space points of one measurement lie on one surface, hence in one group
  std::uint32_t nSharedInGroup[Math::Station::nGroups]{};
  for (std::uint32_t i = 0u; i < a.nHits; ++i) {
    const std::uint32_t rowA{rowOfA(i)};
    for (std::uint32_t j = 0u; j < b.nHits; ++j) {
      if (overlaps(rowA, rowOfB(j))) {
        ++nSharedInGroup[hits.station[rowA]];
        break;
      }
    }
  }
  std::uint32_t nHitsA[Math::Station::nGroups]{};
  std::uint32_t nHitsB[Math::Station::nGroups]{};
  for (std::uint32_t i = 0u; i < a.nHits; ++i) {
    ++nHitsA[hits.station[rowOfA(i)]];
  }
  for (std::uint32_t j = 0u; j < b.nHits; ++j) {
    ++nHitsB[hits.station[rowOfB(j)]];
  }
  std::uint32_t nSharedHits{0u};
  std::uint32_t nSharedGroups{0u};
  std::uint32_t nCommonGroups{0u};
  for (std::uint32_t g = 0u; g < Math::Station::nGroups; ++g) {
    if (nHitsA[g] == 0u || nHitsB[g] == 0u) {
      continue;
    }
    ++nCommonGroups;
    nSharedHits += nSharedInGroup[g];
    // A group is shared when most hits of the smaller pattern in it are shared
    const std::uint32_t smaller{nHitsA[g] < nHitsB[g] ? nHitsA[g] : nHitsB[g]};
    if (2u * nSharedInGroup[g] > smaller) {
      ++nSharedGroups;
    }
  }
  const std::uint32_t minHits{a.nBendingLayers() < b.nBendingLayers()
                                  ? a.nBendingLayers()
                                  : b.nBendingLayers()};
  const std::uint32_t groupCut{cfg.minGroups < nCommonGroups ? cfg.minGroups
                                                             : nCommonGroups};
  return 2u * nSharedHits >= minHits && nSharedGroups >= groupCut;
}

/// Operations on the patterns of one seed: line tests, hit insertion, fits,
/// cuts and overlap checks
struct SeedBuilder {
  const CudaHitPayloadArrays& hits;
  const CudaPatternFinderConfig& cfg;
  const std::uint32_t* rows;
  const std::uint16_t* globLayers;
  const std::uint8_t* infos;
  std::uint32_t maxHits;
  std::uint8_t overflow{0u};

  __device__ std::uint16_t* hitsOf(const Pattern& pat) {
    return reinterpret_cast<std::uint16_t*>(
        const_cast<char*>(reinterpret_cast<const char*>(&pat)) +
        sizeof(Pattern));
  }
  __device__ RealHit hit(std::uint16_t pos) const {
    return Math::loadHit<Real>(hits, rows[pos]);
  }
  __device__ Math::BeamspotT<Real> beamspot() const {
    return Math::convert<Real>(cfg.beamspot);
  }
  __device__ std::uint8_t group(std::uint16_t pos) const {
    return infos[pos] & CandidateInfo::groupMask;
  }
  __device__ std::uint16_t globLayer(std::uint16_t pos) const {
    return globLayers[pos];
  }
  __device__ bool measuresPhi(std::uint16_t pos) const {
    return (infos[pos] & CandidateInfo::measuresPhi) != 0u;
  }
  __device__ bool isPrecision(std::uint16_t pos) const {
    return (infos[pos] & CandidateInfo::precision) != 0u;
  }

  /// Copies in 8-byte words; slots are 8-byte aligned
  __device__ void copy(const Pattern& from, Pattern& to) {
    if (&from == &to) {
      return;
    }
    const std::size_t bytes{sizeof(Pattern) +
                            from.nHits * sizeof(std::uint16_t)};
    const auto* src = reinterpret_cast<const std::uint64_t*>(&from);
    auto* dst = reinterpret_cast<std::uint64_t*>(&to);
    for (std::size_t w = 0u; w < (bytes + 7u) / 8u; ++w) {
      dst[w] = src[w];
    }
  }

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
  __device__ std::uint32_t countGroups(const Pattern& pat, bool onlyGood) {
    return ::countGroups(hits, pat.nHits, rowsOf(pat),
                         onlyGood ? cfg.minGroupLayers : 1u);
  }

  /// Refits the bending plane from the phi hits, group by group in insertion
  /// order, and records how many there are
  __device__ void updateBendPlane(Pattern& pat) {
    pat.needUpdate = (pat.needUpdate & ~Pattern::kBend) | Pattern::kLine;
    std::uint32_t nPhi{0u};
    const std::uint16_t* phiPosAll{hitsOf(pat)};
    for (std::uint16_t i = 0u; i < pat.nHits; ++i) {
      if (measuresPhi(phiPosAll[i])) {
        ++nPhi;
      }
    }
    pat.nPhiLayers = static_cast<std::uint8_t>(nPhi);
    const std::uint16_t* hitPos{hitsOf(pat)};
    // List up to maxListed phi hits once; with more, scan all hits per group
    constexpr std::uint32_t maxListed{32u};
    std::uint16_t phiPos[maxListed];
    std::uint8_t phiGroup[maxListed];
    std::uint32_t nListed{0u};
    if (nPhi <= maxListed) {
      for (std::uint16_t i = 0u; i < pat.nHits; ++i) {
        const std::uint16_t pos{hitPos[i]};
        if (measuresPhi(pos)) {
          phiPos[nListed] = pos;
          phiGroup[nListed] = group(pos);
          ++nListed;
        }
      }
    }
    const bool listed{nPhi <= maxListed};
    auto forEachPhiHit = [&](auto&& fn) {
      for (std::uint32_t g = 0u; g < Math::Station::nGroups; ++g) {
        if (listed) {
          for (std::uint32_t k = 0u; k < nListed; ++k) {
            if (phiGroup[k] == g) {
              const RealHit h{hit(phiPos[k])};
              fn(h);
            }
          }
          continue;
        }
        for (std::uint16_t i = 0u; i < pat.nHits; ++i) {
          const std::uint16_t pos{hitPos[i]};
          if (group(pos) == g && measuresPhi(pos)) {
            const RealHit h{hit(pos)};
            fn(h);
          }
        }
      }
    };
    pat.plane = Math::fitBendPlane(pat.nPhiLayers, pat.plane.phi, pat.sector,
                                   beamspot(), forEachPhiHit);
  }

  __device__ static std::uint32_t layerDistance(std::uint16_t a,
                                                std::uint16_t b) {
    return a > b ? static_cast<std::uint32_t>(a - b)
                 : static_cast<std::uint32_t>(b - a);
  }

  /// The anchor becomes the hit closest in layer to refHit outside refHit's
  /// group. With only one group it is the hit furthest from refHit.
  __device__ void moveLineAnchorHit(Pattern& pat, std::uint16_t refHit) {
    pat.needUpdate = (pat.needUpdate & ~Pattern::kAnchor) | Pattern::kLine;
    const std::uint16_t* hitPos{hitsOf(pat)};
    const std::uint16_t refLayer{globLayer(refHit)};
    if (countGroups(pat, false) < 2u) {
      std::uint16_t best{hitPos[0]};
      std::uint32_t bestDist{0u};
      for (std::uint16_t i = 0u; i < pat.nHits; ++i) {
        const std::uint32_t dist{layerDistance(globLayer(hitPos[i]), refLayer)};
        if (dist > bestDist) {
          bestDist = dist;
          best = hitPos[i];
        }
      }
      pat.lineAnchorHit = best;
      return;
    }
    const std::uint8_t refGroup{group(refHit)};
    std::uint32_t bestDist{UINT_MAX};
    for (std::uint32_t g = 0u; g < Math::Station::nGroups; ++g) {
      if (g == refGroup) {
        continue;
      }
      for (std::uint16_t i = 0u; i < pat.nHits; ++i) {
        if (group(hitPos[i]) != g) {
          continue;
        }
        const std::uint32_t dist{layerDistance(globLayer(hitPos[i]), refLayer)};
        if (dist < bestDist) {
          bestDist = dist;
          pat.lineAnchorHit = hitPos[i];
        }
        if (dist == 1u) {
          return;
        }
      }
    }
  }

  __device__ void updateLineParameters(Pattern& pat) {
    pat.needUpdate &= ~Pattern::kLine;
    pat.lineEndHit = pat.lastInsertedHit;
    const Math::LineT<Real> line{Math::lineThroughHits(
        Math::projectToPhiPlane(hit(pat.lineAnchorHit), pat.plane.normal,
                                pat.plane.offset),
        Math::projectToPhiPlane(hit(pat.lineEndHit), pat.plane.normal,
                                pat.plane.offset),
        cfg.minHitDistance4Line, beamspot())};
    pat.linePos = line.position;
    pat.lineDir = line.direction;
    pat.leverArm = line.leverArm;
    pat.useBeamspot = line.useBeamspot;
  }

  /// Applies the updates booked by addHit. Entering a group that has no hit
  /// yet draws the line across the previous group instead of the old anchor.
  __device__ void syncState(Pattern& pat, bool searchInNewGroup) {
    if ((pat.needUpdate & Pattern::kBend) != 0u) {
      updateBendPlane(pat);
    }
    if ((pat.needUpdate & Pattern::kAnchor) != 0u) {
      pat.lineAnchorHit = pat.prevLayerHit;
      pat.needUpdate = (pat.needUpdate & ~Pattern::kAnchor) | Pattern::kLine;
    }
    const std::uint8_t lastGroup{group(pat.lastInsertedHit)};
    if (searchInNewGroup && lastGroup != group(pat.lineAnchorHit)) {
      const std::uint16_t* hitPos{hitsOf(pat)};
      std::uint16_t furthest{pat.lastInsertedHit};
      std::uint32_t bestDist{0u};
      for (std::uint16_t i = 0u; i < pat.nHits; ++i) {
        if (group(hitPos[i]) != lastGroup) {
          continue;
        }
        const std::uint32_t dist{layerDistance(globLayer(hitPos[i]),
                                               globLayer(pat.lastInsertedHit))};
        if (dist > bestDist) {
          bestDist = dist;
          furthest = hitPos[i];
        }
      }
      const RealVector3 delta{
          Math::projectToPhiPlane(hit(furthest), pat.plane.normal,
                                  pat.plane.offset) -
          Math::projectToPhiPlane(hit(pat.lastInsertedHit), pat.plane.normal,
                                  pat.plane.offset)};
      if (furthest != pat.lastInsertedHit &&
          Math::dot(delta, delta) >=
              Math::square(static_cast<Real>(cfg.minHitDistance4Line))) {
        pat.lineAnchorHit = furthest;
        pat.needUpdate |= Pattern::kLine;
      } else if (globLayer(pat.lineAnchorHit) >
                 globLayer(pat.lastInsertedHit)) {
        pat.lineAnchorHit = pat.lastInsertedHit;
        pat.needUpdate |= Pattern::kLine;
      }
    }
    if ((pat.needUpdate & Pattern::kLine) != 0u) {
      updateLineParameters(pat);
    }
  }

  __device__ bool isPhiCompatible(const Pattern& pat, const RealHit& h) {
    if (pat.nPhiLayers > 0u) {
      return !(std::abs(Math::angleToBendPlane(pat.plane, h.position)) >
               static_cast<Real>(cfg.phiTolerance));
    }
    return Math::ExpandedSector::inside(pat.sector, Math::phi(h.position));
  }

  __device__ LineTest checkLine(Pattern& pat, std::uint16_t test) {
    return checkLine(pat, test, hit(test));
  }
  __device__ LineTest checkLine(Pattern& pat, std::uint16_t test,
                                const RealHit& testHit) {
    if ((testHit.flags & CudaHitFlags::measuresPhi) != 0u &&
        !isPhiCompatible(pat, testHit)) {
      return LineTest{};
    }
    auto makeResult = [&](Decision decision) {
      const Math::LineResidualT<Real> res{
          Math::lineResidual(Math::LineT<Real>{pat.linePos, pat.lineDir,
                                               pat.leverArm, pat.useBeamspot},
                             pat.plane, pat.nPhiLayers, hit(pat.lineAnchorHit),
                             hit(pat.lineEndHit), testHit, beamspot())};
      LineTest out{res.residual, res.sigma, res.nDof, Decision::eRejectHit};
      const bool searchInNewGroup{testHit.station !=
                                  group(pat.lastInsertedHit)};
      const bool searchInNewLayer{globLayer(test) !=
                                  globLayer(pat.lastInsertedHit)};
      Real accWindow{
          static_cast<Real>(res.nDof == 1u ? cfg.nResidualSigma : cfg.cut2Dof) *
          res.sigma};
      if (pat.useBeamspot || searchInNewGroup ||
          (testHit.station != group(pat.prevLayerHit) && !searchInNewLayer)) {
        accWindow *= static_cast<Real>(cfg.loosenAcceptanceFactor);
      }
      if (res.residual < accWindow) {
        out.decision = decision;
      }
      return out;
    };
    if (globLayer(test) != globLayer(pat.lastInsertedHit)) {
      return makeResult(Decision::eAddHit);
    }
    if (test == pat.lastInsertedHit) {
      return LineTest{};
    }
    if (globLayer(pat.lineAnchorHit) == globLayer(pat.lastInsertedHit)) {
      return LineTest{};
    }
    return makeResult(Decision::eBranchPattern);
  }

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
      pat.needUpdate |= Pattern::kBend;
    }
    const bool isNewGroup{group(test) != group(pat.lastInsertedHit)};
    pat.prevLayerHit = pat.lastInsertedHit;
    pat.lastInsertedHit = test;

    pat.sumNormResidual2 += Math::square(res.residual / res.sigma);
    pat.nDoF += res.nDof;
    pat.lastResSigma = res.sigma;
    pat.lastResidual = res.residual;
    pat.lastNDoF = res.nDof;
    if (isNewGroup) {
      pat.needUpdate |= Pattern::kAnchor;
    }
    pat.needUpdate |= Pattern::kLine;
  }

  __device__ void makeSeedPattern(Pattern& pat, std::uint16_t seedPos,
                                  std::int8_t sector) {
    pat = Pattern{};
    pat.lastInsertedHit = seedPos;
    pat.prevLayerHit = seedPos;
    pat.lineAnchorHit = seedPos;
    pat.lineEndHit = seedPos;
    pat.patTheta = Math::theta(hit(seedPos).position);
    pat.sector = sector;
    hitsOf(pat)[0] = seedPos;
    pat.nHits = 1u;
    if (isPrecision(seedPos)) {
      ++pat.nPrecisionLayers;
    } else {
      ++pat.nTriggerLayers;
    }
    // The sector plane stands until the first sync counts the phi hits
    const Real sectorPhi{Math::ExpandedSector::phi<Real>(sector)};
    pat.plane.phi = sectorPhi;
    pat.plane.offset = 0;
    pat.plane.normal =
        RealVector3{-std::sin(sectorPhi), std::cos(sectorPhi), 0};
    pat.plane.covPhiPhi =
        Math::square(Math::ExpandedSector::size<Real>(sector)) / Real{3};
    pat.plane.covSS = 0;
    pat.plane.covPhiS = 0;
    if (measuresPhi(seedPos)) {
      pat.needUpdate |= Pattern::kBend;
    }
    pat.needUpdate |= Pattern::kLine;
  }

  __device__ bool passPatternCuts(const Pattern& pat) {
    return pat.nTriggerLayers >= cfg.minStripEtaLayers &&
           pat.nPrecisionLayers >= cfg.minPrecisionLayers &&
           !(pat.meanResidual2() > static_cast<Real>(cfg.meanNormRes2Cut)) &&
           countGroups(pat, true) >= cfg.minGroups;
  }
};

/// Hit references become indices into the pattern's own hit list
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
  out.sumNormResidual2[slot] = pat.sumNormResidual2;
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
  out.flags[slot] = (pat.useBeamspot ? CudaPatternFlags::useBeamspot : 0u) |
                    ((pat.needUpdate & Pattern::kLine) != 0u
                         ? CudaPatternFlags::needLineUpdate
                         : 0u);
  out.status[slot] = CudaPatternStatus::ok;
  for (std::uint16_t i = 0u; i < pat.nHits; ++i) {
    out.hitIndex[out.hitSlot(slot, i)] = builder.rows[hitPos[i]];
    out.globLayer[out.hitSlot(slot, i)] = builder.globLayers[hitPos[i]];
  }
}

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
  to.sumNormResidual2[toSlot] = from.sumNormResidual2[fromSlot];
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

__device__ PatternInfo recordInfo(const CudaPatternRecordArrays& r,
                                  std::uint32_t slot) {
  return PatternInfo{
      .sector = r.sector[slot],
      .patTheta = static_cast<Real>(r.patTheta[slot]),
      .patPhi = static_cast<Real>(r.patPhi[slot]),
      .patPhiOffset = static_cast<Real>(r.patPhiOffset[slot]),
      .anchorRadius = static_cast<Real>(r.anchorRadius[slot]),
      .nPhiLayers = r.nPhiLayers[slot],
      .nPrecisionLayers = r.nPrecisionLayers[slot],
      .nTriggerLayers = r.nTriggerLayers[slot],
      .meanResidual2 =
          static_cast<Real>(r.sumNormResidual2[slot]) /
          (r.nDoF[slot] > 1u ? static_cast<Real>(r.nDoF[slot]) : Real{1}),
      .nHits = r.nHits[slot]};
}

constexpr unsigned fullMask = 0xFFFFFFFFu;
constexpr unsigned lanesPerWarp = 32u;
constexpr unsigned warpsPerBlock = 4u;

/// Priority order: more good groups, more bending layers, more precision
/// layers, smaller mean squared normalised residual, then smaller index. The
/// index makes the order total, so overlap resolution has a unique result.
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

__device__ std::uint32_t nthSetBit(std::uint32_t mask, std::uint32_t k) {
  for (std::uint32_t i = 0u; i < k; ++i) {
    mask &= mask - 1u;
  }
  return static_cast<std::uint32_t>(__ffs(static_cast<int>(mask)) - 1);
}

/// Warp-wide exclusive prefix sum of count(i) into out; returns the total
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

/// Warp-wide bitonic sort of items[0, n), first the items for which
/// before(a, b) holds
template <typename T, typename Before>
__device__ void warpSort(T* items, std::uint32_t n, Before before,
                         std::uint32_t lane) {
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
        if (before(items[partner], items[t])) {
          const T swap{items[t]};
          items[t] = items[partner];
          items[partner] = swap;
        }
      }
      __syncwarp();
    }
  }
}

constexpr std::uint8_t flagDropped = 1u << 0u;
constexpr std::uint8_t flagNormalHit = 1u << 1u;
/// Not selected when memory ran short
constexpr std::uint8_t flagNoCarry = 1u << 2u;

constexpr std::uint32_t emptySlot = UINT_MAX;

constexpr std::uint8_t undecided = 0u;
constexpr std::uint8_t kept = 1u;
constexpr std::uint8_t dropped = 2u;

/// One block that all seeds allocate from through a shared counter. Memory
/// above patternLimit is reserved for selecting the best patterns when a
/// layer runs out of room.
struct Arena {
  char* base{nullptr};
  unsigned long long* used{nullptr};
  unsigned long long size{0u};
  unsigned long long patternLimit{0u};
};

/// With partial, takes what is left below limit if less than want
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

struct SeedResult {
  const char* region{nullptr};
  const std::uint32_t* keptSlots{nullptr};
  std::uint32_t nKept{0u};
  std::uint32_t stride{0u};
  std::uint32_t maxHits{0u};
};

struct EventCounts {
  unsigned int nPatterns{0u};
  unsigned int maxHits{0u};
  unsigned int largestBeam{0u};
  unsigned int overflow{0u};
};

/// The priority order as one integer, larger being better
__device__ __forceinline__ std::uint64_t priorityKey(std::uint32_t good,
                                                     std::uint32_t bending,
                                                     std::uint32_t precision,
                                                     Real meanResidual2) {
  auto clamp = [](std::uint32_t v, std::uint32_t max) {
    return static_cast<std::uint64_t>(v < max ? v : max);
  };
  const float residual{static_cast<float>(meanResidual2)};
  // Non-negative floats order as their bit patterns; NaN ranks last
  const std::uint64_t residualRank{
      residual >= 0.f ? ~static_cast<std::uint64_t>(__float_as_uint(residual)) &
                            0xFFFFFFFFull
                      : 0u};
  return (clamp(good, 0xFu) << 52u) | (clamp(bending, 0xFFu) << 44u) |
         (clamp(precision, 0xFFu) << 36u) | (residualRank << 4u);
}

struct Candidate {
  std::uint64_t key{0u};
  std::uint32_t order{0u};
  std::uint32_t pad{0u};
};

/// The pattern building of one seed by one warp. Patterns live in two
/// regions that alternate between layers, grown by doubling.
// The line tests of each pattern slot double as the scratch of pruning and
// overlap resolution
static_assert(lanesPerWarp * sizeof(LineTest) >=
              11u * sizeof(std::uint32_t));
static_assert(lanesPerWarp * sizeof(LineTest) >=
              sizeof(PatternInfo) + 2u * sizeof(std::uint32_t));

struct WarpSeed {
  SeedBuilder b;
  Arena arena;
  std::uint32_t stride;
  std::uint32_t maxChunks;
  RealHit* chunkHits;
  std::uint32_t lane;

  /// Named fields: an array indexed at run time would live in local memory
  char* region0{nullptr};
  char* region1{nullptr};
  std::uint32_t regionCap0{0u};
  std::uint32_t regionCap1{0u};
  /// Per-pattern scratch; the masks are indexed [pattern][chunk]
  std::uint32_t scratchCap{0u};
  std::uint32_t* acceptMask{nullptr};
  std::uint32_t* lowMask{nullptr};
  std::uint32_t* offsets{nullptr};
  std::uint8_t* flags{nullptr};
  std::uint8_t* state{nullptr};
  /// [pattern][chunk][hit], reused when the child is built
  LineTest* tests{nullptr};
  std::uint32_t largestBeam{0u};
  /// Families of the patterns being extended: one forward, then one per
  /// forward pattern
  std::uint32_t nFamilies{1u};

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

  /// Room for count patterns in region r, keeping the scratch of the nLive
  /// patterns being extended. Returns how many fit.
  __device__ __forceinline__ std::uint32_t reserve(std::uint32_t r,
                                                   std::uint32_t count,
                                                   std::uint32_t nLive) {
    const std::uint32_t cap{capOf(r)};
    if (count <= cap && count <= scratchCap) {
      return count;
    }
    std::uint32_t target{count > 2u * cap ? count : 2u * cap};
    target = target > lanesPerWarp ? target : lanesPerWarp;
    if (scratchCap < target) {
      const unsigned long long maskBytes{4ull * target * maxChunks};
      const unsigned long long testBytes{sizeof(LineTest) * target * maxChunks *
                                         lanesPerWarp};
      const unsigned long long bytes{testBytes + 2ull * maskBytes +
                                     4ull * target + 2ull * target};
      unsigned long long got{0u};
      char* block{take(bytes, arena.patternLimit, false, got)};
      if (block != nullptr) {
        auto* newTests = reinterpret_cast<LineTest*>(block);
        block += testBytes;
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
        for (std::uint32_t k = lane; k < nLive * maxChunks * lanesPerWarp;
             k += lanesPerWarp) {
          newTests[k] = tests[k];
        }
        for (std::uint32_t k = lane; k < nLive; k += lanesPerWarp) {
          newFlags[k] = flags[k];
        }
        __syncwarp();
        tests = newTests;
        acceptMask = newAccept;
        lowMask = newLow;
        offsets = newOffsets;
        flags = newFlags;
        state = newFlags + target;
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
        (parent.sumNormResidual2 + Math::square(res.residual / res.sigma)) /
            (parent.nDoF + res.nDof));
  }

  /// Keeps the `room` best children and carries of a layer when they do not
  /// all fit; ties are broken by output order
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
      // No memory to rank them: drop the last ones in output order
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
    warpSort(
        candidates, total,
        [](const Candidate& a, const Candidate& c) {
          return a.key > c.key || (a.key == c.key && a.order < c.order);
        },
        lane);
    for (std::uint32_t r = room + lane; r < total; r += lanesPerWarp) {
      drop(candidates[r].order);
    }
    __syncwarp();
  }

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

    // 1. Prune. The line tests of the layer are not written yet, so their
    // memory holds the per-family counters and a hash table of the best
    // pattern of each family and last hit
    std::uint32_t nSlots{2u * lanesPerWarp};
    while (nSlots < 2u * n) {
      nSlots <<= 1u;
    }
    auto* slotKey = reinterpret_cast<std::uint32_t*>(tests);
    std::uint32_t* slotBest{slotKey + nSlots};
    std::uint32_t* familySize{slotBest + nSlots};
    std::uint32_t* familyMissed{familySize + nFamilies};
    std::uint32_t* familyLong{familyMissed + nFamilies};
    for (std::uint32_t k = lane; k < nSlots; k += lanesPerWarp) {
      slotKey[k] = emptySlot;
      slotBest[k] = emptySlot;
    }
    for (std::uint32_t f = lane; f < nFamilies; f += lanesPerWarp) {
      familySize[f] = 0u;
      familyMissed[f] = UINT_MAX;
      familyLong[f] = 0u;
    }
    __syncwarp();
    auto slotOf = [&](const Pattern& pat, bool insert) {
      const std::uint32_t key{(static_cast<std::uint32_t>(pat.family) << 16u) |
                              pat.lastInsertedHit};
      std::uint32_t s{(key * 2654435761u) >> (__clz(nSlots) + 1u)};
      while (true) {
        const std::uint32_t held{insert ? atomicCAS(&slotKey[s], emptySlot, key)
                                        : slotKey[s]};
        if (held == key || held == emptySlot) {
          return s;
        }
        s = (s + 1u) & (nSlots - 1u);
      }
    };
    for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
      const Pattern& pat{pattern(from, i)};
      if (pat.isOverlap) {
        continue;
      }
      atomicAdd(&familySize[pat.family], 1u);
      atomicMin(&familyMissed[pat.family], missedLayers(pat));
      if (pat.nBendingLayers() > 2u) {
        atomicOr(&familyLong[pat.family], 1u);
      }
      const std::uint32_t s{slotOf(pat, true)};
      const PatternInfo info{b.info(pat)};
      std::uint32_t best{atomicCAS(&slotBest[s], emptySlot, i)};
      while (best != emptySlot &&
             outranks(info, 0u, i, b.info(pattern(from, best)), 0u, best)) {
        const std::uint32_t held{atomicCAS(&slotBest[s], best, i)};
        if (held == best) {
          break;
        }
        best = held;
      }
    }
    __syncwarp();
    for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
      const Pattern& pat{pattern(from, i)};
      if (pat.isOverlap) {
        flags[i] = flagDropped;
        continue;
      }
      const std::uint32_t minMissed{familyMissed[pat.family]};
      const std::uint32_t maxMissed{b.cfg.maxMissLayersInGroup > minMissed
                                        ? b.cfg.maxMissLayersInGroup
                                        : minMissed};
      bool drop{b.group(pat.lastInsertedHit) == layerGroup &&
                missedLayers(pat) > maxMissed};
      if (!drop && familySize[pat.family] > 1u &&
          familyLong[pat.family] != 0u) {
        drop = slotBest[slotOf(pat, false)] != i;
      }
      flags[i] = drop ? flagDropped : 0u;
    }
    __syncwarp();
    for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
      if ((flags[i] & flagDropped) == 0u) {
        Pattern& pat{pattern(from, i)};
        b.syncState(pat, b.group(pat.lastInsertedHit) != layerGroup);
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
        if (res.decision != Decision::eAddHit) {
          continue;
        }
        tests[(i * maxChunks + c) * lanesPerWarp + h] = res;
        atomicOr(&acceptMask[i * maxChunks + c], 1u << h);
        if (res.sigma > static_cast<Real>(b.cfg.lowConfidenceResSigma) &&
            res.residual / res.sigma > Real{2}) {
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

    // 3. Make room for all children, or keep the best that fit
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

    // 4. Grow: one child per accepted hit, then the patterns that carry on
    std::uint32_t nOut{0u};
    for (std::uint32_t c = 0u; c < nChunks; ++c) {
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
        // The child has its parent's line, so the parent's test applies
        b.addHit(child, test, tests[(lo * maxChunks + c) * lanesPerWarp + h]);
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

  /// Goes through the patterns best first and keeps each one that no kept
  /// pattern overlaps; with perFamily, only patterns of one family are
  /// compared. The line tests are not in use, so their memory holds the
  /// summaries, the order and the kept patterns.
  __device__ __forceinline__ void resolveOverlaps(std::uint32_t r,
                                                  std::uint32_t n,
                                                  bool perFamily) {
    auto* infos = reinterpret_cast<PatternInfo*>(tests);
    auto* order = reinterpret_cast<std::uint32_t*>(infos + n);
    std::uint32_t* keptList{order + n};
    std::uint32_t* good{offsets};
    for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
      const Pattern& pat{pattern(r, i)};
      state[i] = 0u;
      if (!pat.isOverlap) {
        infos[i] = b.info(pat);
        good[i] = b.countGroups(pat, true);
      }
    }
    const std::uint32_t nLive{warpExclusiveScan(
        n, [&](std::uint32_t i) { return pattern(r, i).isOverlap ? 0u : 1u; },
        keptList, lane)};
    for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
      if (!pattern(r, i).isOverlap) {
        order[keptList[i]] = i;
      }
    }
    __syncwarp();
    auto family = [&](std::uint32_t i) { return pattern(r, i).family; };
    warpSort(
        order, nLive,
        [&](std::uint32_t a, std::uint32_t c) {
          if (perFamily && family(a) != family(c)) {
            return family(a) < family(c);
          }
          return outranks(infos[a], good[a], a, infos[c], good[c], c);
        },
        lane);
    auto overlaps = [&](std::uint32_t upper, std::uint32_t lower) {
      return ::areOverlapping(b.hits, b.cfg, infos[upper],
                              b.rowsOf(pattern(r, upper)), infos[lower],
                              b.rowsOf(pattern(r, lower)));
    };

    // 32 at a time: against the kept patterns, then against the better ones
    // of the same 32, decided in order
    std::uint32_t nKept{0u};
    for (std::uint32_t base = 0u; base < nLive; base += lanesPerWarp) {
      const bool has{base + lane < nLive};
      const std::uint32_t me{has ? order[base + lane] : 0u};
      bool blocked{false};
      for (std::uint32_t k = nKept; has && k > 0u && !blocked; --k) {
        const std::uint32_t other{keptList[k - 1u]};
        if (perFamily && family(other) != family(me)) {
          break;
        }
        blocked = overlaps(other, me);
      }
      std::uint32_t earlier{0u};
      for (std::uint32_t j = 0u; j < lanesPerWarp; ++j) {
        const std::uint32_t other{__shfl_sync(fullMask, me, j)};
        if (has && !blocked && j < lane &&
            (!perFamily || family(other) == family(me)) &&
            overlaps(other, me)) {
          earlier |= 1u << j;
        }
      }
      const unsigned open{__ballot_sync(fullMask, has && !blocked)};
      unsigned keep{0u};
      for (std::uint32_t j = 0u; j < lanesPerWarp; ++j) {
        const unsigned beaten{__shfl_sync(fullMask, earlier, j)};
        if (((open >> j) & 1u) != 0u && (beaten & keep) == 0u) {
          keep |= 1u << j;
        }
      }
      if (((keep >> lane) & 1u) != 0u) {
        keptList[nKept + __popc(keep & ((1u << lane) - 1u))] = me;
        state[me] = 1u;
      }
      nKept += __popc(keep);
      __syncwarp();
    }
    for (std::uint32_t i = lane; i < n; i += lanesPerWarp) {
      if (state[i] == 0u) {
        pattern(r, i).isOverlap = true;
      }
    }
    __syncwarp();
  }

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

    // Forward: the layers after the seed layer
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
    // Restart every forward pattern at the seed, then resolve overlaps.
    // The core fits the restarted line before comparing the patterns.
    nFamilies = n;
    for (std::uint32_t f = lane; f < n; f += lanesPerWarp) {
      Pattern& start{pattern(current, f)};
      start.family = static_cast<std::uint16_t>(f);
      if (start.isOverlap) {
        continue;
      }
      start.lastInsertedHit = seedPos;
      b.moveLineAnchorHit(start, seedPos);
      b.syncState(start, false);
    }
    __syncwarp();
    if (n > 1u) {
      resolveOverlaps(current, n, false);
    }

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

    // Quality cuts, then overlaps within each family, then across families
    for (std::uint32_t k = lane; k < n; k += lanesPerWarp) {
      Pattern& pat{pattern(current, k)};
      pat.isOverlap = pat.isOverlap || !b.passPatternCuts(pat);
    }
    __syncwarp();
    for (std::uint32_t k = lane; k < n; k += lanesPerWarp) {
      Pattern& pat{pattern(current, k)};
      if (!pat.isOverlap) {
        b.syncState(pat, false);
      }
    }
    __syncwarp();
    if (n > 1u) {
      resolveOverlaps(current, n, true);
      resolveOverlaps(current, n, false);
    }

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

__device__ SeedBuilder seedBuilder(const CudaHitPayloadArrays& hits,
                                   const CudaCandidateArrays& lists,
                                   const CudaPatternFinderConfig& cfg,
                                   std::uint32_t seed, std::uint32_t maxHits) {
  return SeedBuilder{.hits = hits,
                     .cfg = cfg,
                     .rows = lists.candidateIndices + lists.begin(seed),
                     .globLayers = lists.candidateGlobLayer + lists.begin(seed),
                     .infos = lists.candidateInfo + lists.begin(seed),
                     .maxHits = maxHits};
}

__global__ void buildPatternsKernel(CudaHitPayloadArrays hits,
                                    CudaCandidateArrays lists,
                                    CudaPatternFinderConfig cfg, Arena arena,
                                    SeedResult* results, EventCounts* counts) {
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
  // At most one hit per layer of the list
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

  // Raw storage: shared memory variables cannot have constructors
  __shared__ alignas(RealHit) unsigned char
      chunkStorage[warpsPerBlock * lanesPerWarp * sizeof(RealHit)];
  RealHit* chunkHits{reinterpret_cast<RealHit*>(chunkStorage) +
                     (threadIdx.x / lanesPerWarp) * lanesPerWarp};
  WarpSeed warp{.b = seedBuilder(hits, lists, cfg, seed, maxHits),
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

/// Allocated on the device once the seeds are built, so the host does not
/// wait for their number
struct EventBuffers {
  CudaPatternRecordArrays event{};
  CudaPatternRecordArrays compacted{};
  std::uint32_t* good{nullptr};
  std::uint32_t* links{nullptr};
  std::uint8_t* state{nullptr};
  std::uint8_t* nextState{nullptr};
  std::uint32_t* firstOfSeed{nullptr};
  std::uint32_t* seedOf{nullptr};
  std::uint32_t* seedLinks{nullptr};
  char* compactedBase{nullptr};
  unsigned long long compactedBytes{0u};
  std::uint32_t nPatterns{0u};
  std::uint32_t wordsPerRow{0u};
  std::uint32_t nKept{0u};
};

/// With a null base, only advances cursor
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
  take(r.sumNormResidual2, n);
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

/// If the buffers do not fit, the event keeps no patterns and gets the
/// eventPatterns overflow bit
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
    const unsigned long long compactedStart{cursor};
    carveRecords(out.compacted, base, cursor, n, maxHits);
    out.compactedBytes = cursor - compactedStart;
    if (base != nullptr) {
      out.compactedBase = base + compactedStart;
    }
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
    take(out.seedOf, n);
    take(out.seedLinks, static_cast<unsigned long long>(n) * words);
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
  __shared__ std::uint32_t warpTotals[32];
  __shared__ std::uint32_t nTotal;
  const std::uint32_t lane{threadIdx.x % lanesPerWarp};
  const std::uint32_t warp{threadIdx.x / lanesPerWarp};
  const std::uint32_t nWarps{blockDim.x / lanesPerWarp};
  if (threadIdx.x == 0u) {
    nTotal = 0u;
  }
  __syncthreads();
  for (std::uint32_t base = 0u; base < lists.nSeeds; base += blockDim.x) {
    const std::uint32_t s{base + threadIdx.x};
    const std::uint32_t value{s < lists.nSeeds ? results[s].nKept : 0u};
    std::uint32_t sum{value};
    for (std::uint32_t d = 1u; d < lanesPerWarp; d <<= 1u) {
      const std::uint32_t other{__shfl_up_sync(fullMask, sum, d)};
      if (lane >= d) {
        sum += other;
      }
    }
    if (lane == lanesPerWarp - 1u) {
      warpTotals[warp] = sum;
    }
    __syncthreads();
    std::uint32_t before{nTotal};
    for (std::uint32_t w = 0u; w < warp; ++w) {
      before += warpTotals[w];
    }
    if (s < lists.nSeeds) {
      firstOfSeed[s] = before + sum - value;
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
      for (std::uint32_t w = 0u; w < nWarps; ++w) {
        nTotal += warpTotals[w];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0u) {
    event.nPatterns[0] = nTotal;
    event.overflow[0] = static_cast<std::uint8_t>(counts->overflow);
  }
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
    buffers->seedOf[t] = lo;
    SeedBuilder builder{seedBuilder(hits, lists, cfg, lo, result.maxHits)};
    const auto& pat = *reinterpret_cast<const Pattern*>(
        result.region +
        static_cast<std::size_t>(result.keptSlots[t - firstOfSeed[lo]]) *
            result.stride);
    writePattern(builder, pat, builder.hitsOf(pat), event, t);
  }
}

/// Seed link: the higher-ranked pattern holds the other's seed hit and they
/// come from different seeds
__device__ void linkPair(const CudaHitPayloadArrays& hits,
                         const CudaCandidateArrays& lists,
                         const CudaPatternFinderConfig& cfg,
                         const EventBuffers& buffers, std::uint32_t i,
                         std::uint32_t j) {
  const CudaPatternRecordArrays& event{buffers.event};
  const std::uint32_t* good{buffers.good};
  const std::uint32_t wordsPerRow{buffers.wordsPerRow};
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
  const std::uint32_t bit{1u << (upper % 32u)};
  const std::size_t word{static_cast<std::size_t>(lower) * wordsPerRow +
                         upper / 32u};
  if (areOverlapping(hits, cfg, iFirst ? infoI : infoJ, rowsOf(upper),
                     iFirst ? infoJ : infoI, rowsOf(lower))) {
    atomicOr(&buffers.links[word], bit);
  }
  const std::uint32_t seedLower{buffers.seedOf[lower]};
  if (buffers.seedOf[upper] != seedLower) {
    // A middle-layer seed is counted in half the theta window of an
    // inner or outer seed, matching SeedSelector::thetaSearchWindow
    const std::uint8_t seedLayer{
        Math::layerOf(hits.station[lists.seedHitIndex[seedLower]])};
    const Real thetaCut{seedLayer == Math::Layer::Inner ||
                                seedLayer == Math::Layer::Outer
                            ? static_cast<Real>(cfg.maxThetaSizeOverlap)
                            : static_cast<Real>(0.5 * cfg.maxThetaSizeOverlap)};
    if (std::abs(infoI.patTheta - infoJ.patTheta) > thetaCut) {
      return;
    }
    const std::uint32_t seedRow{lists.seedHitIndex[seedLower]};
    const auto upperRows = rowsOf(upper);
    for (std::uint32_t k = 0u; k < event.nHits[upper]; ++k) {
      if (upperRows(k) == seedRow) {
        atomicOr(&buffers.seedLinks[word], bit);
        break;
      }
    }
  }
}

/// Bit j of row i is set if pattern j ranks above pattern i and overlaps it
/// (overlap links) or holds its seed hit (seed links). Rows must be zeroed.
__global__ void linkEventPatternsKernel(CudaHitPayloadArrays hits,
                                        CudaCandidateArrays lists,
                                        CudaPatternFinderConfig cfg,
                                        const EventBuffers* buffers) {
  const std::uint32_t n{buffers->nPatterns};
  const CudaPatternRecordArrays event{buffers->event};
  const std::uint64_t nPairs{static_cast<std::uint64_t>(n) * n};
  for (std::uint64_t pair =
           blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
       pair < nPairs;
       pair += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
    const auto i = static_cast<std::uint32_t>(pair / n);
    const auto j = static_cast<std::uint32_t>(pair % n);
    if (j <= i ||
        !Math::ExpandedSector::isNeighbour(event.sector[i], event.sector[j]) ||
        std::abs(static_cast<Real>(event.patTheta[i]) -
                 static_cast<Real>(event.patTheta[j])) >
            static_cast<Real>(cfg.maxThetaSizeOverlap)) {
      continue;
    }
    linkPair(hits, lists, cfg, *buffers, i, j);
  }
}

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
    buffers->seedLinks[w] = 0u;
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

/// Rounds over the links. A pattern is also dropped if its seed hit is held
/// by maxSeedAttempts higher-ranked kept patterns of other seeds. The result
/// equals accepting the patterns one by one in priority order.
__global__ void resolveEventOverlapsKernel(const EventBuffers* buffers,
                                           std::uint32_t maxSeedAttempts) {
  const std::uint32_t n{buffers->nPatterns};
  const CudaPatternRecordArrays event{buffers->event};
  const std::uint32_t* links{buffers->links};
  const std::uint32_t* seedLinks{buffers->seedLinks};
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
      std::uint32_t nHolding{0u};
      for (std::uint32_t w = 0u; w < wordsPerRow && !isDropped; ++w) {
        std::uint32_t word{seedLinks[i * wordsPerRow + w]};
        while (word != 0u && !isDropped) {
          const std::uint32_t j{32u * w +
                                static_cast<std::uint32_t>(__ffs(word) - 1)};
          word &= word - 1u;
          nHolding += state[j] == kept ? 1u : 0u;
          isDropped = nHolding >= maxSeedAttempts;
          isBlocked = isBlocked || state[j] == undecided;
        }
      }
      isDropped = isDropped || nHolding >= maxSeedAttempts;
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

/// One warp per seed: group counts, bitonic sort by layer, and global layer
/// numbers by prefix sum
__global__ void prepareSeedsKernel(CudaHitPayloadArrays hits,
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
  std::uint8_t* info{lists.candidateInfo + lists.begin(seed)};
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

  // The padding beyond n is never read
  std::uint32_t padded{1u};
  while (padded < n) {
    padded <<= 1u;
  }
  auto sortRows = [&](auto before) {
    for (std::uint32_t k = 2u; k <= padded; k <<= 1u) {
      for (std::uint32_t j = k / 2u; j > 0u; j >>= 1u) {
        for (std::uint32_t t = lane; t < padded; t += lanesPerWarp) {
          const std::uint32_t partner{j == k / 2u ? t ^ (k - 1u) : t ^ j};
          if (partner <= t || partner >= n) {
            continue;
          }
          const std::uint32_t rowT{rows[t]};
          const std::uint32_t rowP{rows[partner]};
          if (before(rowP, rowT)) {
            rows[t] = rowP;
            rows[partner] = rowT;
          }
        }
        __syncwarp();
      }
    }
  };
  // The layer order is not transitive for hits of different sectors, so the
  // rows are put in a fixed order first; otherwise the result would depend on
  // the order of the candidate list
  sortRows([](std::uint32_t a, std::uint32_t b) { return a < b; });
  sortRows([&](std::uint32_t a, std::uint32_t b) {
    return Math::layerSorter(loadSortHit(hits, a), loadSortHit(hits, b));
  });

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
      const std::uint8_t flags{hits.flags[rows[i]]};
      info[i] = static_cast<std::uint8_t>(
          (hits.station[rows[i]] & CandidateInfo::groupMask) |
          ((flags & CudaHitFlags::measuresPhi) != 0u
               ? CandidateInfo::measuresPhi
               : 0u) |
          ((flags & CudaHitFlags::precision) != 0u ? CandidateInfo::precision
                                                   : 0u));
    }
    carry += __shfl_sync(fullMask, sum, lanesPerWarp - 1u);
  }
  if (carry + 1u < minLayers) {
    setStatus(CudaSeedStatus::tooFewLayers);
    return;
  }

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

__device__ Math::Vector3 rotate(const double* rotation, double x, double y,
                                double z) {
  return {rotation[0] * x + rotation[3] * y + rotation[6] * z,
          rotation[1] * x + rotation[4] * y + rotation[7] * z,
          rotation[2] * x + rotation[5] * y + rotation[8] * z};
}
__device__ Math::Vector3 column(const double* rotation, std::uint32_t c) {
  return {rotation[3u * c], rotation[3u * c + 1u], rotation[3u * c + 2u]};
}
__device__ Math::Vector3 cross(const Math::Vector3& a, const Math::Vector3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

/// Bit layout of MuonSpacePoint::MuonId, see MuonSpacePoint.cpp
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
constexpr std::uint32_t mdt = 0u;
constexpr std::uint32_t tgc = 3u;
constexpr std::uint32_t sTgc = 4u;
constexpr std::uint32_t mm = 5u;
/// StationName order: BIS, BIL, BMS, BML, BOS, BOL, BEE, EIS, EIL, EMS, EML,
/// EOS, EOL, EES, EEL
__device__ std::uint8_t station(std::uint32_t stationName) {
  constexpr std::uint8_t map[15]{0u, 0u, 1u, 1u, 2u, 2u, 3u, 4u,
                                 4u, 5u, 5u, 6u, 6u, 7u, 7u};
  return stationName < 15u ? map[stationName] : 0xFFu;
}
}  // namespace MuonIdBits

__global__ void buildHitPayloadsKernel(CudaMuonSpacePointArrays sp,
                                       CudaUsedHitArrays used,
                                       CudaSurfaceTableArrays surfaces,
                                       CudaHitPayloadArrays out,
                                       std::uint32_t* nMissingSurfaces) {
  const std::uint32_t r{blockIdx.x * blockDim.x + threadIdx.x};
  if (r >= used.nHits) {
    return;
  }
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
    // Only the eta strips of two-dimensional TGC measurements can be
    // non-orthogonal to their phi strips
    bool nonOrthogonal{false};
    double stripAngle{0.};
    if (measEta && measPhi && !straw && tech == MuonIdBits::tgc) {
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
    Math::Vector3 sensorDir{};
    if (straw) {
      sensorDir = column(rot, 2u);
    } else if (nonOrthogonal) {
      sensorDir = -std::sin(stripAngle) * column(rot, 1u) +
                  std::cos(stripAngle) * column(rot, 0u);
    } else {
      sensorDir = column(rot, 1u);
    }
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

/// Candidate search over a sorted array: one (sector, theta) entry per hit and
/// expanded sector, sorted; a seed's window [min, max) is found by two binary
/// searches in its sector
namespace Search {

constexpr std::uint32_t projections = 3u;
constexpr std::uint8_t noSector = 0xFFu;
constexpr double degree = Math::pi / 180.;

/// Transverse direction at phi + 90 degrees
__device__ Math::Vector3 normalDir(int sector) {
  const double phi{Math::ExpandedSector::phi(sector) + 90. * degree};
  const double theta{90. * degree};
  const double sinTheta{std::sin(theta)};
  return {std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, std::cos(theta)};
}

__device__ std::uint32_t bucketOf(const CudaMuonSpacePointArrays& sp,
                                  std::uint32_t nBuckets, std::uint32_t r) {
  std::uint32_t bucket{0u};
  std::uint32_t pastBucket{nBuckets};
  while (pastBucket - bucket > 1u) {
    const std::uint32_t mid{(bucket + pastBucket) / 2u};
    if (sp.bucketStart[mid] <= r) {
      bucket = mid;
    } else {
      pastBucket = mid;
    }
  }
  return bucket;
}

/// Negative if the hit does not seed
__device__ double seedWindow(std::uint32_t raw, const CudaSeedSelection& sel) {
  const std::uint8_t layer{
      Math::layerOf(MuonIdBits::station(MuonIdBits::stationName(raw)))};
  const bool straw{MuonIdBits::technology(raw) == MuonIdBits::mdt};
  if (((sel.seedingLayers >> layer) & 1u) == 0u ||
      (straw && !sel.seedFromMdt)) {
    return -1.;
  }
  return (layer == Math::Layer::Inner || layer == Math::Layer::Outer)
             ? sel.thetaSearchWindow
             : 0.5 * sel.thetaSearchWindow;
}

struct Buffers {
  // Entry index is row * 3 + projection
  std::uint64_t* thetaKey;
  std::uint8_t* sector;
  double* theta;
  std::uint32_t* entry;
  std::uint64_t* thetaKeySorted;
  std::uint32_t* entrySorted;
  std::uint8_t* sectorSorted;
  std::uint8_t* sectorSorted2;
  std::uint32_t* entrySorted2;
  std::uint32_t* isSeed;
  std::uint32_t* seedSlot;
  std::uint32_t* count;
  std::uint32_t* offset;
  std::uint32_t* low;
  // Per sector, [first, last) of its entries
  std::uint32_t* segment;
  std::uint32_t* totals;
  std::uint32_t nEntries;
};

__global__ void entriesKernel(CudaMuonSpacePointArrays sp, CudaSearchArrays in,
                              Buffers buf) {
  const std::uint32_t r{blockIdx.x * blockDim.x + threadIdx.x};
  if (r >= in.nHits) {
    return;
  }
  const std::uint32_t b{bucketOf(sp, in.nBuckets, r)};
  const double* rot{in.bucketRotation + 9u * b};
  const double* t{in.bucketTranslation + 3u * b};
  const double lx{sp.localPositionX[r]};
  const double ly{sp.localPositionY[r]};
  const double lz{sp.localPositionZ[r]};
  // Without fused multiply-add, to match the host
  auto row = [&](int i) {
    return __dadd_rn(
        __dadd_rn(__dadd_rn(__dmul_rn(rot[i], lx), __dmul_rn(rot[3 + i], ly)),
                  __dmul_rn(rot[6 + i], lz)),
        t[i]);
  };
  const Math::Vector3 pos{row(0), row(1), row(2)};
  in.positionX[r] = pos.x;
  in.positionY[r] = pos.y;
  in.positionZ[r] = pos.z;

  const std::uint32_t raw{sp.muonId[r]};
  const bool measPhi{MuonIdBits::measuresPhi(raw)};
  const auto msSector =
      static_cast<unsigned>(MuonIdBits::sector(sp.muonId[sp.bucketStart[b]]));
  const double phi{std::atan2(pos.y, pos.x)};
  const double r2{__dadd_rn(__dmul_rn(pos.x, pos.x), __dmul_rn(pos.y, pos.y))};
  for (int proj = -1; proj <= 1; ++proj) {
    const std::uint32_t e{projections * r +
                          static_cast<std::uint32_t>(proj + 1)};
    const unsigned sector{(2u * msSector + static_cast<unsigned>(proj)) % 32u};
    buf.entry[e] = e;
    // A phi-measuring hit enters an overlap sector only if its phi is inside
    if (proj != 0 && measPhi &&
        !Math::ExpandedSector::inside(static_cast<int>(sector), phi)) {
      buf.thetaKey[e] = ~0ull;
      buf.sector[e] = noSector;
      buf.theta[e] = 0.;
      continue;
    }
    // Other hits are projected onto the sector plane: sqrt(r^2 - (pos.n)^2)
    double projR{std::sqrt(r2)};
    if (!measPhi) {
      const Math::Vector3 n{normalDir(static_cast<int>(sector))};
      const double d{
          __dadd_rn(__dadd_rn(__dmul_rn(pos.x, n.x), __dmul_rn(pos.y, n.y)),
                    __dmul_rn(pos.z, n.z))};
      const double rest{__dadd_rn(r2, -__dmul_rn(d, d))};
      projR = std::sqrt(rest > 0. ? rest : 0.);
    }
    const double theta{std::atan2(projR, pos.z)};
    buf.theta[e] = theta;
    // Non-negative, so the bits order as the values
    buf.thetaKey[e] = static_cast<std::uint64_t>(__double_as_longlong(theta));
    buf.sector[e] = static_cast<std::uint8_t>(sector);
  }
}

__global__ void gatherSectorsKernel(Buffers buf) {
  const std::uint32_t i{blockIdx.x * blockDim.x + threadIdx.x};
  if (i < buf.nEntries) {
    buf.sectorSorted[i] = buf.sector[buf.entrySorted[i]];
  }
}

__global__ void segmentsKernel(Buffers buf) {
  const std::uint32_t i{blockIdx.x * blockDim.x + threadIdx.x};
  if (i >= buf.nEntries) {
    return;
  }
  const std::uint8_t s{buf.sectorSorted2[i]};
  if (s == noSector) {
    return;
  }
  if (i == 0u || buf.sectorSorted2[i - 1u] != s) {
    buf.segment[2u * s] = i;
  }
  if (i + 1u == buf.nEntries || buf.sectorSorted2[i + 1u] != s) {
    buf.segment[2u * s + 1u] = i + 1u;
  }
}

__global__ void windowsKernel(CudaMuonSpacePointArrays sp,
                              CudaSeedSelection sel, Buffers buf) {
  const std::uint32_t i{blockIdx.x * blockDim.x + threadIdx.x};
  if (i >= buf.nEntries) {
    return;
  }
  buf.isSeed[i] = 0u;
  buf.count[i] = 0u;
  const std::uint8_t s{buf.sectorSorted2[i]};
  if (s == noSector) {
    return;
  }
  const std::uint32_t e{buf.entrySorted2[i]};
  const double window{seedWindow(sp.muonId[e / projections], sel)};
  if (window < 0.) {
    return;
  }
  const double t{buf.theta[e]};
  const double lowEdge{t - window / 2.};
  const double highEdge{t + window / 2.};
  auto lowerBound = [&](double value) {
    std::uint32_t lo{buf.segment[2u * s]};
    std::uint32_t hi{buf.segment[2u * s + 1u]};
    while (lo < hi) {
      const std::uint32_t mid{(lo + hi) / 2u};
      if (buf.theta[buf.entrySorted2[mid]] < value) {
        lo = mid + 1u;
      } else {
        hi = mid;
      }
    }
    return lo;
  };
  const std::uint32_t lo{lowerBound(lowEdge)};
  const std::uint32_t hi{lowerBound(highEdge)};
  buf.isSeed[i] = 1u;
  buf.low[i] = lo;
  buf.count[i] = hi - lo;
}

__global__ void totalsKernel(Buffers buf) {
  if (threadIdx.x == 0u && blockIdx.x == 0u) {
    const std::uint32_t last{buf.nEntries - 1u};
    buf.totals[0] = buf.seedSlot[last] + buf.isSeed[last];
    buf.totals[1] = buf.offset[last] + buf.count[last];
  }
}

__global__ void fillListsKernel(Buffers buf, CudaCandidateArrays lists) {
  const std::uint32_t i{blockIdx.x * blockDim.x + threadIdx.x};
  if (i >= buf.nEntries || buf.isSeed[i] == 0u) {
    return;
  }
  const std::uint32_t seed{buf.seedSlot[i]};
  const std::uint32_t e{buf.entrySorted2[i]};
  lists.seedHitIndex[seed] = e / projections;
  lists.seedSector[seed] = static_cast<double>(buf.sectorSorted2[i]);
  lists.seedTheta[seed] = buf.theta[e];
  const std::uint32_t first{buf.offset[i]};
  lists.candidateOffsets[seed] = first;
  if (seed + 1u == lists.nSeeds) {
    lists.candidateOffsets[seed + 1u] = first + buf.count[i];
  }
  for (std::uint32_t k = 0u; k < buf.count[i]; ++k) {
    lists.candidateIndices[first + k] =
        buf.entrySorted2[buf.low[i] + k] / projections;
  }
}

}  // namespace Search

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

void prepareCudaPatternFinder(std::size_t arenaBytes, cudaStream_t stream) {
  int device{0};
  ACTS_CUDA_CHECK(cudaGetDevice(&device));
  cudaMemPool_t pool{};
  ACTS_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, device));
  std::uint64_t keepAll{std::numeric_limits<std::uint64_t>::max()};
  ACTS_CUDA_CHECK(
      cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keepAll));

  // With lazy module loading a kernel is loaded at its first launch; querying
  // its attributes loads it now
  cudaFuncAttributes attributes{};
  for (const void* kernel :
       {reinterpret_cast<const void*>(&buildHitPayloadsKernel),
        reinterpret_cast<const void*>(&Search::entriesKernel),
        reinterpret_cast<const void*>(&Search::gatherSectorsKernel),
        reinterpret_cast<const void*>(&Search::segmentsKernel),
        reinterpret_cast<const void*>(&Search::windowsKernel),
        reinterpret_cast<const void*>(&Search::totalsKernel),
        reinterpret_cast<const void*>(&Search::fillListsKernel),
        reinterpret_cast<const void*>(&prepareSeedsKernel),
        reinterpret_cast<const void*>(&buildPatternsKernel),
        reinterpret_cast<const void*>(&carveEventKernel),
        reinterpret_cast<const void*>(&appendPatternsKernel),
        reinterpret_cast<const void*>(&countEventGroupsKernel),
        reinterpret_cast<const void*>(&linkEventPatternsKernel),
        reinterpret_cast<const void*>(&resolveEventOverlapsKernel),
        reinterpret_cast<const void*>(&compactEventPatternsKernel)}) {
    ACTS_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  }

  // The CUB kernels have no handle to query: run the search's sorts and scans
  // once on a single element
  {
    std::uint64_t* keys{nullptr};
    std::uint8_t* sectors{nullptr};
    std::uint32_t* values{nullptr};
    char* temp{nullptr};
    ACTS_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&keys),
                                    2u * sizeof(std::uint64_t), stream));
    ACTS_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&sectors),
                                    2u * sizeof(std::uint8_t), stream));
    ACTS_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&values),
                                    4u * sizeof(std::uint32_t), stream));
    ACTS_CUDA_CHECK(cudaMemsetAsync(keys, 0, 2u * sizeof(std::uint64_t), stream));
    ACTS_CUDA_CHECK(cudaMemsetAsync(sectors, 0, 2u * sizeof(std::uint8_t), stream));
    ACTS_CUDA_CHECK(cudaMemsetAsync(values, 0, 4u * sizeof(std::uint32_t), stream));
    std::size_t sortKeyBytes{0u};
    std::size_t sortSectorBytes{0u};
    std::size_t scanBytes{0u};
    ACTS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sortKeyBytes, keys, keys + 1, values, values + 1, 1, 0, 64,
        stream));
    ACTS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sortSectorBytes, sectors, sectors + 1, values, values + 1, 1, 0,
        8, stream));
    ACTS_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, values,
                                                  values + 2, 1, stream));
    std::size_t tempBytes{std::max({sortKeyBytes, sortSectorBytes, scanBytes})};
    ACTS_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&temp),
                                    std::max<std::size_t>(tempBytes, 1u), stream));
    ACTS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        temp, tempBytes, keys, keys + 1, values, values + 1, 1, 0, 64, stream));
    ACTS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        temp, tempBytes, sectors, sectors + 1, values, values + 1, 1, 0, 8,
        stream));
    ACTS_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp, tempBytes, values,
                                                  values + 2, 1, stream));
    for (void* ptr : {static_cast<void*>(keys), static_cast<void*>(sectors),
                      static_cast<void*>(values), static_cast<void*>(temp)}) {
      ACTS_CUDA_CHECK(cudaFreeAsync(ptr, stream));
    }
  }

  // Taking the arena once leaves it in the pool for the events
  if (arenaBytes > 0u) {
    void* arena{nullptr};
    ACTS_CUDA_CHECK(cudaMallocAsync(&arena, arenaBytes, stream));
    ACTS_CUDA_CHECK(cudaFreeAsync(arena, stream));
  }
  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));
}

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

void findCudaCandidates(const CudaMuonSpacePointArrays& spacePoints,
                        const CudaSearchArrays& search,
                        const CudaSeedSelection& selection,
                        CudaCandidateListContainer& lists,
                        CudaPinnedBuffer& readback, cudaStream_t stream) {
  lists = CudaCandidateListContainer{};
  if (search.nHits == 0u) {
    return;
  }
  std::vector<void*> blocks{};
  auto allocate = [&](auto*& ptr, std::size_t count) {
    ACTS_CUDA_CHECK(cudaMallocAsync(
        reinterpret_cast<void**>(&ptr),
        std::max<std::size_t>(count, 1u) * sizeof(*ptr), stream));
    blocks.push_back(ptr);
  };
  Search::Buffers buf{};
  const std::uint32_t n{search.nHits * Search::projections};
  buf.nEntries = n;
  allocate(buf.thetaKey, n);
  allocate(buf.sector, n);
  allocate(buf.theta, n);
  allocate(buf.entry, n);
  allocate(buf.thetaKeySorted, n);
  allocate(buf.entrySorted, n);
  allocate(buf.sectorSorted, n);
  allocate(buf.sectorSorted2, n);
  allocate(buf.entrySorted2, n);
  allocate(buf.isSeed, n);
  allocate(buf.seedSlot, n);
  allocate(buf.count, n);
  allocate(buf.offset, n);
  allocate(buf.low, n);
  allocate(buf.segment, 2u * 32u);
  allocate(buf.totals, 2u);
  std::size_t sortThetaBytes{0u};
  std::size_t sortSectorBytes{0u};
  std::size_t scanBytes{0u};
  ACTS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, sortThetaBytes, buf.thetaKey, buf.thetaKeySorted, buf.entry,
      buf.entrySorted, n, 0, 64, stream));
  ACTS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, sortSectorBytes, buf.sectorSorted, buf.sectorSorted2,
      buf.entrySorted, buf.entrySorted2, n, 0, 8, stream));
  ACTS_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, buf.isSeed,
                                                buf.seedSlot, n, stream));
  std::size_t tempBytes{std::max({sortThetaBytes, sortSectorBytes, scanBytes})};
  char* temp{nullptr};
  allocate(temp, tempBytes);
  ACTS_CUDA_CHECK(cudaMemsetAsync(buf.segment, 0,
                                  2u * 32u * sizeof(std::uint32_t), stream));

  constexpr unsigned blockSize{256u};
  const unsigned hitBlocks{(search.nHits + blockSize - 1u) / blockSize};
  const unsigned entryBlocks{(n + blockSize - 1u) / blockSize};
  Search::entriesKernel<<<hitBlocks, blockSize, 0, stream>>>(spacePoints,
                                                             search, buf);
  ACTS_CUDA_CHECK(cudaGetLastError());
  // The radix sort is stable: sorting by theta, then by sector, keeps the
  // theta order within a sector
  ACTS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      temp, tempBytes, buf.thetaKey, buf.thetaKeySorted, buf.entry,
      buf.entrySorted, n, 0, 64, stream));
  Search::gatherSectorsKernel<<<entryBlocks, blockSize, 0, stream>>>(buf);
  ACTS_CUDA_CHECK(cudaGetLastError());
  ACTS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      temp, tempBytes, buf.sectorSorted, buf.sectorSorted2, buf.entrySorted,
      buf.entrySorted2, n, 0, 8, stream));
  Search::segmentsKernel<<<entryBlocks, blockSize, 0, stream>>>(buf);
  ACTS_CUDA_CHECK(cudaGetLastError());
  Search::windowsKernel<<<entryBlocks, blockSize, 0, stream>>>(spacePoints,
                                                               selection, buf);
  ACTS_CUDA_CHECK(cudaGetLastError());
  ACTS_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp, tempBytes, buf.isSeed,
                                                buf.seedSlot, n, stream));
  ACTS_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp, tempBytes, buf.count,
                                                buf.offset, n, stream));
  Search::totalsKernel<<<1, 1, 0, stream>>>(buf);
  ACTS_CUDA_CHECK(cudaGetLastError());

  auto* totals = reinterpret_cast<std::uint32_t*>(
      readback.reserve(2u * sizeof(std::uint32_t)));
  ACTS_CUDA_CHECK(cudaMemcpyAsync(totals, buf.totals,
                                  2u * sizeof(std::uint32_t),
                                  cudaMemcpyDeviceToHost, stream));
  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));
  const std::uint32_t nSeeds{totals[0]};
  const std::uint32_t nCandidates{totals[1]};
  if (nSeeds > 0u) {
    lists.allocateDevice(nSeeds, nCandidates, stream);
    Search::fillListsKernel<<<entryBlocks, blockSize, 0, stream>>>(
        buf, lists.deviceArrays());
    ACTS_CUDA_CHECK(cudaGetLastError());
  }
  for (void* ptr : blocks) {
    ACTS_CUDA_CHECK(cudaFreeAsync(ptr, stream));
  }
}

void prepareCudaSeeds(const CudaHitPayloadContainer& hits,
                      CudaCandidateListContainer& lists,
                      const CudaPatternFinderConfig& cfg, cudaStream_t stream) {
  if (!hits.isOnDevice() || !lists.isOnDevice()) {
    throw std::invalid_argument(
        "prepareCudaSeeds: the hits and the candidate lists must be on the "
        "device");
  }
  const auto nSeeds = static_cast<std::uint32_t>(lists.nSeeds());
  if (nSeeds == 0u) {
    return;
  }
  prepareSeedsKernel<<<(nSeeds + warpsPerBlock - 1u) / warpsPerBlock,
                       warpsPerBlock * lanesPerWarp, 0, stream>>>(
      hits.deviceArrays(), lists.deviceArrays(), cfg);
  ACTS_CUDA_CHECK(cudaGetLastError());
}

CudaPatternRecordContainer findCudaPatternsInEta(
    const CudaHitPayloadContainer& hits, CudaCandidateListContainer& lists,
    const CudaPatternFinderConfig& cfg, const CudaPatternFinderSizes& sizes,
    cudaStream_t stream, CudaPatternFinderTiming* timing,
    CudaPinnedBuffer* readback) {
  if (sizes.workspaceBytes == 0u) {
    throw std::invalid_argument(
        "findCudaPatternsInEta: the workspace has to have some memory");
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
  prepareCudaSeeds(hits, lists, cfg, stream);

  Arena arena{};
  arena.size = sizes.workspaceBytes & ~7ull;
  // A sixteenth is reserved for selecting the best patterns
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
    buildPatternsKernel<<<nWarpBlocks, warpsPerBlock * lanesPerWarp, 0,
                          stream>>>(hits.deviceArrays(), lists.deviceArrays(),
                                    cfg, arena, results, counts);
    ACTS_CUDA_CHECK(cudaGetLastError());
  }
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
  linkEventPatternsKernel<<<512, 256, 0, stream>>>(
      hits.deviceArrays(), lists.deviceArrays(), cfg, eventBuffers);
  ACTS_CUDA_CHECK(cudaGetLastError());
  resolveEventOverlapsKernel<<<1, 1024, 0, stream>>>(eventBuffers,
                                                     cfg.maxSeedAttempts);
  ACTS_CUDA_CHECK(cudaGetLastError());
  // Only the kept patterns are copied back
  compactEventPatternsKernel<<<1, 256, 0, stream>>>(eventBuffers);
  ACTS_CUDA_CHECK(cudaGetLastError());
  mark(1u);
  CudaPinnedBuffer ownReadback{};
  CudaPinnedBuffer& pinned{readback != nullptr ? *readback : ownReadback};
  constexpr std::size_t countsAt{(sizeof(EventBuffers) + 15u) & ~15u};
  char* small{pinned.reserve(countsAt + sizeof(EventCounts))};
  ACTS_CUDA_CHECK(cudaMemcpyAsync(small, eventBuffers, sizeof(EventBuffers),
                                  cudaMemcpyDeviceToHost, stream));
  ACTS_CUDA_CHECK(cudaMemcpyAsync(small + countsAt, counts, sizeof(EventCounts),
                                  cudaMemcpyDeviceToHost, stream));
  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));
  EventBuffers ready{};
  EventCounts totals{};
  std::memcpy(&ready, small, sizeof(EventBuffers));
  std::memcpy(&totals, small + countsAt, sizeof(EventCounts));
  const std::uint32_t maxHits{std::max(totals.maxHits, 1u)};
  CudaPatternRecordContainer kept{1u, ready.nKept, maxHits};
  if (ready.nKept > 0u) {
    ACTS_CUDA_CHECK(cudaMemcpyAsync(pinned.reserve(ready.compactedBytes),
                                    ready.compactedBase, ready.compactedBytes,
                                    cudaMemcpyDeviceToHost, stream));
  }
  mark(2u);
  release();
  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));
  if (ready.nKept > 0u) {
    kept.copyFromHostCopy(ready.compacted, ready.compactedBase, pinned.data());
  }
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
    timing->largestBeam = totals.largestBeam;
    timing->overflow = overflow;
    timing->nPatternsInEta = nKept[0];
    for (cudaEvent_t& m : marks) {
      ACTS_CUDA_CHECK(cudaEventDestroy(m));
    }
  }
  return kept;
}

}  // namespace ActsExamples
