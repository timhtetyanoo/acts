// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFinding/MuonGlobalPatternFinderDefs.hpp"

#include "Acts/Definitions/Common.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/Utilities/MathHelpers.hpp"
#include "ActsExamples/TrackFinding/MuonGlobalPatternFinderUtils.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace Acts::UnitLiterals;

namespace {

/// @brief Gradient of the azimuthal coordinate
/// @param pos: Position where the gradient is to be computed
Acts::Vector3 phiGradient(const Acts::Vector3& pos) {
  return Acts::Vector3{-pos.y(), pos.x(), 0.} / pos.perp2();
}

}  // namespace

namespace ActsExamples::MuonGlobalPatternFinderDefs {

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
      phiCov = discCov / position.perp2() +
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
       << hit->position.perp() << " / " << hit->position.phi() / 1_degree
       << ", st: " << station
       << ", loc/glob lay: " << static_cast<int>(hit->locLayer) << "/"
       << static_cast<int>(globLayer);
}

}  // namespace ActsExamples::MuonGlobalPatternFinderDefs
