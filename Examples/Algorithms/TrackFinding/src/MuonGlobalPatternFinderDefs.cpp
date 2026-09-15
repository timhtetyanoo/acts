// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFinding/MuonGlobalPatternFinderDefs.hpp"

#include "Acts/Definitions/Common.hpp"
#include "Acts/Surfaces/detail/PlanarHelper.hpp"
#include "Acts/Utilities/MathHelpers.hpp"
#include "Acts/Utilities/UnitVectors.hpp"
#include "Acts/Utilities/VectorHelpers.hpp"
#include "Acts/Utilities/detail/periodic.hpp"
#include "ActsExamples/TrackFinding/MuonGlobalPatternFinderUtils.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace Acts::UnitLiterals;
using Acts::VectorHelpers::perp;
using Acts::VectorHelpers::phi;
using Acts::VectorHelpers::theta;

namespace ActsExamples::MuonGlobalPatternFinderDefs {

namespace {

/// @brief Gradient of the azimuthal coordinate
/// @param pos: Position where the gradient is to be computed
Acts::Vector3 phiGradient(const Acts::Vector3& pos) {
  return Acts::Vector3{-pos.y(), pos.x(), 0.} / pos.head<2>().squaredNorm();
}

/// @brief Size of an expanded sector in phi
double expandedSectorSize(const MuonExpandedSector& sect) {
  const auto sector1 = static_cast<int>(sect.msSector());
  const auto sector2 = static_cast<int>(sect.adjacentMsSector());
  if (sector1 == sector2) {
    return MuonSectorMapping::sectorSize(sector1);
  }
  // The overlap size is the same for small and large sectors
  return MuonSectorMapping::sectorWidth(sector1) -
         MuonSectorMapping::sectorSize(sector1);
}

/// @brief Convert an angle from radians to degrees
constexpr double inDeg(double angle) {
  return angle / 1._degree;
}

}  // namespace

HitPayload::HitPayload(const MuonSpacePoint* sp,
                       const MuonSpacePointBucket* bucket,
                       const Acts::Transform3& localToGlobal,
                       std::uint8_t locLayer, MuonStationIndex station)
    : position{localToGlobal * sp->localPosition()},
      sp{sp},
      bucket{bucket},
      station{station},
      locLayer{locLayer},
      isStraw{sp->isStraw()},
      isPrecision{MuonGlobalPatternFinderUtils::isPrecisionHit(*sp)},
      measuresPhi{sp->id().measuresPhi()},
      measuresEta{sp->id().measuresEta()} {
  const auto& rotation = localToGlobal.rotation();
  const auto& cov = sp->covariance();
  const double covPhi = cov[Acts::toUnderlying(CovIdx::phiCov)];
  const double covEta = cov[Acts::toUnderlying(CovIdx::etaCov)];

  if (!measuresEta) {
    // Phi-only measurement: we need only the phi covariance
    const Acts::Vector3 phiMeasDir = rotation * sp->toNextSensor();
    phiCov = covPhi * Acts::square(phiMeasDir.dot(phiGradient(position)));
    return;
  }

  nonOrthogonalStrips =
      !isStraw && measuresPhi &&
      std::abs(sp->sensorDirection().dot(sp->toNextSensor())) > Acts::s_epsilon;

  sensorDir =
      nonOrthogonalStrips
          ? Acts::Vector3{rotation *
                          sp->planeNormal().cross(sp->toNextSensor()).normalized()}
          : Acts::Vector3{rotation * sp->sensorDirection()};

  if (isStraw) {
    // For straw hits, the x component of secondaryMeasDir is repurposed to
    // store the transverse covariance of the drift radius
    double& discCov = secondaryMeasDir.x();
    discCov = Acts::square(sp->driftRadius()) + covEta;

    if (measuresPhi) {
      phiCov = discCov / position.head<2>().squaredNorm() +
               Acts::square(sensorDir.dot(phiGradient(position))) *
                   (covPhi - discCov);
    }
    return;
  }
  // Strip hits measuring eta and eventually phi
  secondaryMeasDir = nonOrthogonalStrips ? rotation * sp->sensorDirection()
                                         : rotation * sp->toNextSensor();
  if (!measuresPhi) {
    return;
  }
  const Acts::Vector3 gradPhi = phiGradient(position);
  if (nonOrthogonalStrips) {
    const Acts::Vector3 primaryMeasDir = rotation * sp->toNextSensor();
    phiCov = covEta * Acts::square(primaryMeasDir.dot(gradPhi)) +
             covPhi * Acts::square(secondaryMeasDir.dot(gradPhi));
  } else {
    phiCov = covEta * Acts::square(secondaryMeasDir.dot(gradPhi)) +
             covPhi * Acts::square(sensorDir.dot(gradPhi));
  }
}

double HitPayload::residualVariance(const Acts::Vector3& contractionVector,
                                    bool isProjected) const {
  // If the hit is not projected, the contraction vector is the residual
  // direction
  assert(isProjected ||
         std::abs(contractionVector.norm() - 1.) < Acts::s_epsilon);
  const auto& cov = sp->covariance();
  const double covPhi = cov[Acts::toUnderlying(CovIdx::phiCov)];
  const double covEta = cov[Acts::toUnderlying(CovIdx::etaCov)];

  if (isStraw) {
    const double discCov = secondaryMeasDir.x();
    if (!isProjected) {
      const double vDotRsq = Acts::square(sensorDir.dot(contractionVector));
      return measuresPhi ? discCov * (1. - vDotRsq) + vDotRsq * covPhi
                         : discCov * (1. - vDotRsq);
    }
    // If the hit is projected, the phi measurement, if available, is
    // cancelled by the projection
    return discCov * contractionVector.squaredNorm();
  }
  if (!measuresEta) {
    throw std::runtime_error(
        "HitPayload: phi-only hits are not meant to be used for residual "
        "computation.");
  }
  if (nonOrthogonalStrips) {
    const Acts::Vector3 primaryMeasDir =
        (secondaryMeasDir - sensorDir.dot(secondaryMeasDir) * sensorDir)
            .normalized();
    return covEta * Acts::square(primaryMeasDir.dot(contractionVector)) +
           covPhi * Acts::square(secondaryMeasDir.dot(contractionVector));
  }
  const double etaTerm =
      covEta * Acts::square(secondaryMeasDir.dot(contractionVector));
  return measuresPhi
             ? etaTerm + covPhi * Acts::square(sensorDir.dot(contractionVector))
             : etaTerm;
}

void CandidateHit::print(std::ostream& ostr) const {
  ostr << *sp() << ", glob Z/R/phi: " << hit->position.z() << " / "
       << perp(hit->position) << " / " << inDeg(phi(hit->position))
       << ", st: " << station
       << ", loc/glob lay: " << static_cast<int>(hit->locLayer) << "/"
       << static_cast<int>(globLayer);
}

PatternState::PatternState(const CandidateHit& seed, std::int8_t expSector,
                           const PatternFinderConfig* cfg,
                           const Acts::Logger* logger)
    : cfg{cfg},
      m_logger{logger},
      lastInsertedHit{seed},
      prevLayerHit{seed},
      lineAnchorHit{seed},
      patTheta{theta(seed->position)},
      expSect{expSector} {
  if (cfg == nullptr || logger == nullptr) {
    throw std::invalid_argument(
        "PatternState: the configuration and the logger must be provided");
  }
  hitsPerStation[stationIdx(seed.station)].push_back(seed);

  if (seed->isPrecision) {
    ++nPrecisionLayers;
  } else {
    ++nTriggerLayers;
  }
  if (seed->measuresPhi) {
    ++nPhiLayers;
  }
  updatePatternPhi();
  needLineUpdate = true;
}

LineTestRes PatternState::checkLineComp(const CandidateHit& testHit,
                                        const Acts::Vector3& beamSpot) {
  if (testHit->measuresPhi && !isPhiCompatible(*testHit)) {
    ACTS_VERBOSE("checkLineComp() Test hit phi "
                 << inDeg(phi(testHit->position)) << " not compatible with "
                 << brief(*this));
    return LineTestRes{};
  }
  /// @brief Compute the residual and set the decision if the residual is
  ///        within the acceptance window
  auto makeResult = [&testHit, this](LineTestDecision decision) {
    LineTestRes res = computeLineResidual(testHit);
    double accWindow = cfg->nResidualSigma * res.sigma;
    // Loosen the window when the beamspot is used or when looking for hits in
    // a new station, as the straight line approximation becomes less accurate
    // over large distances
    if (useBeamspot || testHit.station != lastInsertedHit.station ||
        (testHit.station != prevLayerHit.station &&
         testHit.globLayer == lastInsertedHit.globLayer)) {
      accWindow *= 2.;
    }
    if (res.residual < accWindow) {
      res.result = decision;
    }
    return res;
  };

  if (testHit.globLayer != lastInsertedHit.globLayer) {
    updateLineParameters(beamSpot);
    return makeResult(LineTestDecision::eAddHit);
  }
  if (testHit == lastInsertedHit) {
    ACTS_VERBOSE("checkLineComp() Test hit is the last inserted hit - reject.");
    return LineTestRes{};
  }
  if (lineAnchorHit.globLayer == lastInsertedHit.globLayer) {
    ACTS_VERBOSE("checkLineComp() Test hit on the same layer as the seed with "
                 "no prior hits - reject.");
    return LineTestRes{};
  }
  // The line is intentionally not updated: the test hit is an alternative to
  // the last inserted hit, which is not part of the current line
  return makeResult(LineTestDecision::eBranchPattern);
}

LineTestRes PatternState::computeLineResidual(
    const CandidateHit& testHit) const {
  LineTestRes res{};

  // The test hit is projected onto the phi plane only if it does not measure
  // phi or if the pattern has no phi layers. Otherwise, the residual includes
  // the error in the phi direction.
  const bool projectTestHit = !testHit->measuresPhi || nPhiLayers == 0u;
  const Acts::Vector3 testPos =
      projectTestHit ? projToPhiPlane(*testHit) : testHit->position;
  const Acts::Vector3 K = testPos - linePos;
  const double KdotD = K.dot(lineDir);
  const Acts::Vector3 residualVec = K - KdotD * lineDir;
  res.residual = residualVec.norm();
  if (res.residual < Acts::s_epsilon) {
    // A vanishing residual is likely due to a bad topology, reject it
    res.residual = std::numeric_limits<double>::max();
    res.sigma = 0.;
    return res;
  }
  const Acts::Vector3 resDir = residualVec / res.residual;
  // Extrapolation distance along the pattern line in units of the lever arm
  const double alpha = KdotD / leverArm;

  // Derivative of the residual w.r.t. the common phi-plane angle
  double phiPlaneDerivativeAcc{0.};
  // Covariance contributions of the hits to the residual
  double residualCovAcc{0.};

  /// @brief Accumulate the covariance contributions of one hit. Each hit
  ///        contributes with its intrinsic covariance and, if projected, with
  ///        the uncertainty on the phi-plane angle of the pattern.
  ///        For a projected hit, the residual direction is transformed with
  ///        the projection Jacobian J^T * dir, such that
  ///        dir^T * J * cov * J^T * dir = (J^T * dir)^T * cov * (J^T * dir)
  /// @param hit: Hit for which to compute the covariance
  /// @param pos: Projected position of the hit
  /// @param preFactor: Factor, function of alpha, to scale the contributions
  /// @param isProjected: Whether the hit is projected
  auto covarianceTerm = [&](const HitPayload& hit, const Acts::Vector3& pos,
                            double preFactor, bool isProjected) {
    if (!isProjected) {
      residualCovAcc +=
          Acts::square(preFactor) * hit.residualVariance(resDir, false);
      return;
    }
    const double projFactor =
        hit.sensorDir.dot(resDir) / hit.sensorDir.dot(bendPlaneNorm);
    const Acts::Vector3 trfDir = resDir - projFactor * bendPlaneNorm;
    residualCovAcc +=
        Acts::square(preFactor) * hit.residualVariance(trfDir, true);
    phiPlaneDerivativeAcc += preFactor * perp(pos) * projFactor;
  };

  // Contribution of the first line point
  if (useBeamspot) {
    // @note Ported from Athena: the beamspot length & radius enter as variances
    const double covS1 =
        cfg->beamSpotLength * Acts::square(resDir.z()) +
        cfg->beamSpotRadius * (1. - Acts::square(resDir.z()));
    residualCovAcc += Acts::square(alpha - 1.) * covS1;
  } else {
    covarianceTerm(*lineAnchorHit, linePos, alpha - 1., true);
  }
  // Contribution of the second line point
  const Acts::Vector3 pos2 = linePos + leverArm * lineDir;
  covarianceTerm(*lastInsertedHit, pos2, -alpha, true);
  // Contribution of the test hit
  covarianceTerm(*testHit, testPos, 1., projectTestHit);

  res.sigma = std::sqrt(residualCovAcc +
                        Acts::square(phiPlaneDerivativeAcc) * patPhiCov);

  ACTS_VERBOSE("computeLineResidual() "
               << brief(*this) << "\nUse beamspot: " << useBeamspot
               << ", alpha: " << alpha << ", Residual: " << res.residual
               << " +- " << res.sigma << ", linePos R/theta: "
               << perp(linePos) << " / " << inDeg(theta(linePos))
               << ", lineDir theta: " << inDeg(theta(lineDir))
               << ", testPos R/theta/phi: " << perp(testPos) << " / "
               << inDeg(theta(testPos)) << " / " << inDeg(phi(testPos))
               << ", hit pos sigma: " << std::sqrt(residualCovAcc)
               << ", phi plane sigma: "
               << std::abs(phiPlaneDerivativeAcc) * std::sqrt(patPhiCov));
  return res;
}

void PatternState::addHit(const CandidateHit& hit, double residual,
                          double resSigma) {
  hitsPerStation[stationIdx(hit.station)].push_back(hit);

  if (hit->isPrecision) {
    ++nPrecisionLayers;
  } else {
    ++nTriggerLayers;
  }
  if (hit->measuresPhi) {
    ++nPhiLayers;
    updatePatternPhi();
  }

  const bool isNewStation = hit.station != lastInsertedHit.station;
  prevLayerHit = lastInsertedHit;
  lastInsertedHit = hit;

  meanNormResidual2 += Acts::square(residual / resSigma);
  lastResSigma = resSigma;
  lastResidual = residual;

  // If the new hit is in a different station, update the line anchor
  if (isNewStation) {
    moveLineAnchorHit(hit);
  }
  needLineUpdate = true;
}

void PatternState::overWriteHit(const CandidateHit& newHit, double newResidual,
                                double newResSigma) {
  const MuonStationIndex st = newHit.station;
  if (st != lastInsertedHit.station ||
      lastInsertedHit.globLayer != newHit.globLayer) {
    throw std::runtime_error(std::format(
        "PatternState: trying to overwrite a hit in station/layer {}/{} with "
        "another one from station/layer {}/{}",
        toString(lastInsertedHit.station),
        static_cast<int>(lastInsertedHit.globLayer), toString(st),
        static_cast<int>(newHit.globLayer)));
  }
  // Hits of the same type (precision / trigger) are expected to be replaced,
  // since patterns are only branched with compatible hits on the same layer.
  // The exception are sTgc pads replaced by strips on the same layer
  if (lastInsertedHit->isPrecision != newHit->isPrecision) {
    const bool isStgcStrip =
        newHit.sp()->id().technology() ==
            MuonSpacePoint::MuonId::TechField::sTgc &&
        newHit->isPrecision;
    if (!isStgcStrip) {
      std::ostringstream sstr{};
      sstr << "PatternState: trying to overwrite a hit with an incompatible "
              "type\nOld hit: "
           << *lastInsertedHit.sp()
           << ", isPrecision: " << lastInsertedHit->isPrecision
           << "\nNew hit: " << *newHit.sp()
           << ", isPrecision: " << newHit->isPrecision;
      throw std::runtime_error(sstr.str());
    }
    ++nPrecisionLayers;
    --nTriggerLayers;
  }
  auto& stHits = hitsPerStation[stationIdx(st)];
  if (stHits.empty() || stHits.back() != lastInsertedHit) {
    std::ostringstream sstr{};
    sstr << "PatternState: trying to overwrite a hit that is not the last "
            "inserted hit in station/layer "
         << st << "/" << static_cast<int>(lastInsertedHit.globLayer)
         << "\nLast inserted hit: " << *lastInsertedHit.sp();
    throw std::runtime_error(sstr.str());
  }

  bool updatePhi{false};
  if (lastInsertedHit->measuresPhi) {
    --nPhiLayers;
    updatePhi = true;
  }
  if (newHit->measuresPhi) {
    ++nPhiLayers;
    updatePhi = true;
  }

  meanNormResidual2 += Acts::square(newResidual / newResSigma) -
                       Acts::square(lastResidual / lastResSigma);
  lastResSigma = newResSigma;
  lastResidual = newResidual;

  stHits.back() = newHit;
  lastInsertedHit = newHit;

  if (updatePhi) {
    updatePatternPhi();
  }
  needLineUpdate = true;
}

void PatternState::moveLineAnchorHit(const CandidateHit& refHit) {
  // Treat first the special case where we have only one station
  if (nStations(false) < 2u) {
    // The hit search direction has been inverted without finding any hit in
    // other stations beside the initial one. So the anchor is the last hit.
    lineAnchorHit = lastInsertedHit;
    return;
  }
  // Find first the closest station to the reference hit among the pattern
  // stations
  const auto closestSt = std::ranges::min_element(
      hitsPerStation, std::ranges::less{},
      [&refHit](const std::vector<CandidateHit>& hits) {
        if (hits.empty() || hits.front().station == refHit.station) {
          return std::numeric_limits<int>::max();
        }
        return std::abs(hits.front().globLayer - refHit.globLayer);
      });
  // Then find the closest hit in that station to the reference hit
  lineAnchorHit = *std::ranges::min_element(
      *closestSt, std::ranges::less{}, [&refHit](const CandidateHit& hit) {
        return std::abs(hit.globLayer - refHit.globLayer);
      });
}

void PatternState::updateLineParameters(const Acts::Vector3& beamSpot) {
  if (!needLineUpdate) {
    return;
  }
  const Acts::Vector3 pos1 = projToPhiPlane(*lineAnchorHit);
  const Acts::Vector3 pos2 = projToPhiPlane(*lastInsertedHit);
  Acts::Vector3 d = pos2 - pos1;
  leverArm = d.norm();

  // Check whether we have to use the beamspot instead of the anchor hit to
  // draw the line
  useBeamspot = lastInsertedHit.station == lineAnchorHit.station &&
                leverArm < cfg->minHitDistance4Line;
  if (useBeamspot) {
    linePos = beamSpot;
    d = pos2 - beamSpot;
    leverArm = d.norm();
  } else {
    linePos = pos1;
  }
  lineDir = d / leverArm;
  needLineUpdate = false;

  ACTS_VERBOSE("updateLineParameters() Updated --> linePos R/z/theta: "
               << perp(linePos) << " / " << linePos.z() << " / "
               << inDeg(theta(linePos))
               << ", lineDir theta: " << inDeg(theta(lineDir))
               << ", LeverArm: " << leverArm
               << ", Use beamspot: " << useBeamspot);
}

Acts::Vector3 PatternState::projToPhiPlane(const HitPayload& hit) const {
  // The bending plane contains the beam axis, i.e. it has no offset
  return Acts::PlanarHelper::intersectPlane(hit.position, hit.sensorDir,
                                            bendPlaneNorm, 0.)
      .position();
}

void PatternState::updatePatternPhi() {
  if (nPhiLayers == 0u) {
    // Without phi hits, use the central phi of the sector / overlap region,
    // with the variance of a uniform distribution over the expanded sector
    patPhi = MuonSectorMapping::sectorOverlapPhi(
        static_cast<int>(expSect.msSector()),
        static_cast<int>(expSect.adjacentMsSector()));
    patPhiCov = Acts::square(expandedSectorSize(expSect)) / 3.;
    bendPlaneNorm =
        Acts::makeDirectionFromPhiTheta(patPhi + 90._degree, 90._degree);
    ACTS_VERBOSE("updatePatternPhi() No phi hits in the pattern, set pattern "
                 "phi to "
                 << inDeg(patPhi) << " +- " << inDeg(std::sqrt(patPhiCov)));
    return;
  }
  double sumSin{0.};
  double sumCos{0.};
  double sumWeight{0.};
  auto processPhiHit = [&sumSin, &sumCos, &sumWeight](const HitPayload& hit) {
    if (!hit.measuresPhi) {
      return;
    }
    if (hit.phiCov < Acts::s_epsilon) {
      std::ostringstream sstr{};
      sstr << "PatternState: unexpected phi hit with zero variance in the phi "
              "direction: "
           << *hit.sp;
      throw std::runtime_error(sstr.str());
    }
    const double weight = 1. / hit.phiCov;
    const double hitPhi = phi(hit.position);
    sumSin += weight * std::sin(hitPhi);
    sumCos += weight * std::cos(hitPhi);
    sumWeight += weight;
  };
  for (const std::vector<CandidateHit>& hits : hitsPerStation) {
    for (const CandidateHit& hit : hits) {
      processPhiHit(*hit);
    }
  }
  for (const HitPayload& hit : phiOnlyHits) {
    processPhiHit(hit);
  }
  patPhi = std::atan2(sumSin, sumCos);
  patPhiCov = 1. / sumWeight;
  bendPlaneNorm =
      Acts::makeDirectionFromPhiTheta(patPhi + 90._degree, 90._degree);
  ACTS_VERBOSE("updatePatternPhi() Updated pattern phi to "
               << inDeg(patPhi) << " +- " << inDeg(std::sqrt(patPhiCov)));
}

bool PatternState::isPhiCompatible(const HitPayload& hit) const {
  const double testPhi = phi(hit.position);
  if (nPhiLayers > 0u) {
    const double deltaPhiSigma = std::sqrt(patPhiCov + hit.phiCov);
    const double deltaPhi = Acts::detail::radian_sym(patPhi - testPhi);
    if (std::abs(deltaPhi) > cfg->nPhiSigma * deltaPhiSigma) {
      ACTS_VERBOSE("isPhiCompatible() The pattern with phi = "
                   << inDeg(patPhi) << " +- " << inDeg(std::sqrt(patPhiCov))
                   << " is not compatible with the test hit with phi "
                   << inDeg(testPhi) << " +- " << inDeg(std::sqrt(hit.phiCov)));
      return false;
    }
    return true;
  }
  const auto sector1 = static_cast<int>(expSect.msSector());
  const auto sector2 = static_cast<int>(expSect.adjacentMsSector());
  const bool isCompatible =
      MuonSectorMapping::insideSector(sector1, testPhi) &&
      (sector1 == sector2 || MuonSectorMapping::insideSector(sector2, testPhi));
  if (!isCompatible) {
    ACTS_VERBOSE("isPhiCompatible() The test hit with phi = "
                 << inDeg(testPhi) << " is not inside the pattern sectors: "
                 << sector1 << " and " << sector2);
  }
  return isCompatible;
}

bool PatternState::isInPattern(const HitPayload& hit) const {
  const std::vector<CandidateHit>& hits =
      hitsPerStation[stationIdx(hit.station)];
  return std::ranges::any_of(
      hits, [&hit](const CandidateHit& c) { return *c == hit; });
}

std::uint8_t PatternState::nStations(bool onlyGoodStations) const {
  return static_cast<std::uint8_t>(std::ranges::count_if(
      hitsPerStation,
      [this, onlyGoodStations](const std::vector<CandidateHit>& hits) {
        return !hits.empty() &&
               (!onlyGoodStations || hits.size() >= cfg->minStationLayers);
      }));
}

std::uint8_t PatternState::nBendingLayers() const {
  return static_cast<std::uint8_t>(nPrecisionLayers + nTriggerLayers);
}

double PatternState::getMeanResidual2() const {
  if (isFinalized) {
    return meanNormResidual2;
  }
  return meanNormResidual2 / nBendingLayers();
}

std::vector<const MuonSpacePointBucket*> PatternState::getParentBuckets()
    const {
  std::vector<const MuonSpacePointBucket*> buckets{};
  for (const std::vector<CandidateHit>& hits : hitsPerStation) {
    for (const CandidateHit& hit : hits) {
      if (std::ranges::find(buckets, hit->bucket) == buckets.end()) {
        buckets.push_back(hit->bucket);
      }
    }
  }
  return buckets;
}

void PatternState::print(std::ostream& ostr, bool detailedPrint) const {
  ostr << "PatternState Exp Sector: " << static_cast<int>(expSect.sector())
       << ", Theta: " << inDeg(patTheta) << ", Phi: " << inDeg(patPhi)
       << " +- " << inDeg(std::sqrt(patPhiCov))
       << ", nPrec: " << static_cast<int>(nPrecisionLayers)
       << ", nEtaNonPrec: " << static_cast<int>(nTriggerLayers)
       << ", nPhi: " << static_cast<int>(nPhiLayers)
       << ", mean norm res sq: " << getMeanResidual2()
       << ", dirTheta: " << inDeg(theta(lineDir)) << ", Hit per station: \n";
  for (std::size_t st = 0u; st < s_nStations; ++st) {
    const std::vector<CandidateHit>& hits = hitsPerStation[st];
    if (hits.empty()) {
      continue;
    }
    ostr << "  Station "
         << static_cast<MuonStationIndex>(static_cast<std::int8_t>(st))
         << " has " << hits.size() << " hits ";
    if (detailedPrint) {
      ostr << "\n";
      for (const CandidateHit& hit : hits) {
        ostr << "    " << hit << "\n";
      }
    }
  }
  if (!detailedPrint) {
    ostr << "\n    Last hit: " << lastInsertedHit
         << "\n    prevLayerHit: " << prevLayerHit
         << "\n    lineAnchorHit: " << lineAnchorHit;
  }
}

}  // namespace ActsExamples::MuonGlobalPatternFinderDefs
