// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Seeding/detail/CompSpacePointAuxiliaries.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "ActsExamples/EventData/MuonGlobalPatternFinderEvent.hpp"
#include "ActsExamples/EventData/MuonSpacePoint.hpp"

#include <cstdint>
#include <ostream>

/// @brief Internal data structures of the muon global pattern finder
namespace ActsExamples::MuonGlobalPatternFinderDefs {

/// @brief Components of the space point covariance array
enum class CovIdx : std::uint8_t {
  phiCov = Acts::toUnderlying(
      Acts::Experimental::detail::CompSpacePointAuxiliaries::ResidualIdx::
          nonBending),
  etaCov = Acts::toUnderlying(
      Acts::Experimental::detail::CompSpacePointAuxiliaries::ResidualIdx::
          bending),
  timeCov = Acts::toUnderlying(
      Acts::Experimental::detail::CompSpacePointAuxiliaries::ResidualIdx::time)
};

/// @brief Hit information cached for the pattern finding. It holds the global
///        position & sensor directions of the space point together with its
///        azimuthal covariance.
struct HitPayload {
  /// @brief Constructor
  /// @param sp: The space point
  /// @param bucket: The bucket holding the space point
  /// @param localToGlobal: Transform from the space point frame to global
  /// @param locLayer: The layer number in the bucket
  /// @param station: The station index
  explicit HitPayload(const MuonSpacePoint* sp,
                      const MuonSpacePointBucket* bucket,
                      const Acts::Transform3& localToGlobal,
                      std::uint8_t locLayer, MuonStationIndex station);

  /// @brief Hit contribution to the residual variance due to its intrinsic
  ///        position uncertainty
  /// @param contractionVector: The contraction vector to compute the residual
  ///        variance. If the hit is not projected, it is the unit residual
  ///        direction. Otherwise, it's J^T * residualDirection, where J is the
  ///        Jacobian of the projection
  /// @param isProjected: Whether the hit has been projected
  /// @return Residual variance contribution
  double residualVariance(const Acts::Vector3& contractionVector,
                          bool isProjected) const;

  /// @brief Global position
  Acts::Vector3 position{Acts::Vector3::Zero()};
  /// @brief Global sensor direction of the precision measurement
  Acts::Vector3 sensorDir{Acts::Vector3::Zero()};
  /// @brief For strip hits with phi: measurement direction independent of
  ///        sensorDir, which is the eta measurement direction if the strips
  ///        are orthogonal, the phi measurement direction otherwise.
  ///        For straw hits, the x component is repurposed to store the
  ///        transverse covariance of the drift radius.
  Acts::Vector3 secondaryMeasDir{Acts::Vector3::Zero()};
  /// @brief Pointer to the underlying space point
  const MuonSpacePoint* sp{nullptr};
  /// @brief Pointer to the parent bucket
  const MuonSpacePointBucket* bucket{nullptr};
  /// @brief Cached angular covariance [rad^2] of the hit in the phi angle
  double phiCov{0.};
  /// @brief Station index
  MuonStationIndex station{MuonStationIndex::UnDef};
  /// @brief Layer number in the bucket
  std::uint8_t locLayer{0u};
  /// @brief Is the hit a straw
  bool isStraw{false};
  /// @brief Is precision hit
  bool isPrecision{false};
  /// @brief Does the hit measure phi
  bool measuresPhi{false};
  /// @brief Does the hit measure eta
  bool measuresEta{false};
  /// @brief Are the strips non-orthogonal
  bool nonOrthogonalStrips{false};

  /// @brief Equal operator: it compares the underlying space point
  bool operator==(const HitPayload& other) const { return sp == other.sp; }
  /// @brief Arrow operator: it allows to access the underlying space point
  const MuonSpacePoint* operator->() const { return sp; }
  /// @brief Dereference operator: it allows to access the underlying space point
  const MuonSpacePoint& operator*() const { return *sp; }
};

/// @brief Small wrapper for candidate hits used to build patterns. This is
///        needed because the global layer number cannot be defined globally,
///        but it can be computed given a set of hits.
struct CandidateHit {
  /// @brief Pointer to the underlying hit
  const HitPayload* hit{nullptr};
  /// @brief Station index
  MuonStationIndex station{MuonStationIndex::UnDef};
  /// @brief Global measurement layer number
  std::uint8_t globLayer{0u};

  const HitPayload* operator->() const { return hit; }
  const HitPayload& operator*() const { return *hit; }
  const MuonSpacePoint* sp() const { return hit->sp; }
  bool operator==(const CandidateHit& other) const {
    return *hit == *other.hit;
  }
  bool operator==(const HitPayload& other) const { return *hit == other; }

  /// @brief Print the candidate hit
  void print(std::ostream& ostr) const;
  friend std::ostream& operator<<(std::ostream& ostr, const CandidateHit& c) {
    c.print(ostr);
    return ostr;
  }
};

}  // namespace ActsExamples::MuonGlobalPatternFinderDefs
