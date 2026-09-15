// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/EventData/CompositeSpacePoint.hpp"
#include "Acts/Geometry/GeometryContext.hpp"

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace Acts::Experimental {
/// Result of comparing the station / layer of two hits
enum class HitOrdering : std::int8_t {
  /// The first hit is in a lower station / layer than the second one
  eLower = -1,
  /// Both hits are in the same station / layer
  eSame = 0,
  /// The first hit is in a higher station / layer than the second one
  eHigher = 1
};

/// Concept definition of a hit payload. A hit payload wraps a pointer to a
/// space point satisfying the `CompositeSpacePoint` concept and augments it
/// with global quantities
template <typename Payload_t>
concept HitPayload =
    requires(const Payload_t h, const Payload_t other,
             const GeometryContext& gctx) {
      /// Equal operator: it compares the underlying hit
      { h == other } -> std::same_as<bool>;
      /// Pointer (ordinary / smart) to the underlying space point, which
      /// satisfies the CompositeSpacePoint concept
      requires CompositeSpacePointPtr<
          std::remove_cvref_t<decltype(h.spacePoint())>>;
      /// Position of the hit in the global frame
      { h.globalPosition(gctx) } -> std::convertible_to<Vector3>;
      /// Sensor direction in the global frame
      { h.globalSensorDir(gctx) } -> std::convertible_to<Vector3>;
      /// Variance of the residual of the hit
      { h.residualVariance(gctx) } -> std::same_as<double>;
      /// Covariance of the hit in the phi direction
      { h.phiCovariance() } -> std::same_as<double>;
      /// Returns whether the hit is a precision measurement
      { h.isPrecision() } -> std::same_as<bool>;
      /// Compares station of h with respect to station of other
      { compareStation(h, other) } -> std::same_as<HitOrdering>;
      /// Compares layer of h with respect to layer of other
      { compareLayer(h, other) } -> std::same_as<HitOrdering>;
    };

}  // namespace Acts::Experimental
