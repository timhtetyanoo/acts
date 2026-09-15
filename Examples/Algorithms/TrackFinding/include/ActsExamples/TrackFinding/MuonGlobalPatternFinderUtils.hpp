// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "ActsExamples/EventData/MuonSpacePoint.hpp"

#include <cstdint>
#include <vector>

namespace Acts {
class TrackingGeometry;
}  // namespace Acts

/// @brief Space point helpers needed by the muon global pattern finder. They
///        replace the Athena services (MuonIdHelperSvc, SpacePointPerLayerSorter,
///        SpacePointHelpers) that the original implementation relies on.
namespace ActsExamples::MuonGlobalPatternFinderUtils {

/// @brief Default tolerance to consider two space points on the same
///        measurement layer
constexpr double s_layerTolerance = 10. * Acts::UnitConstants::um;

/// @brief Returns whether the space point is a precision hit, i.e. an Mdt, a
///        Micromegas or an sTgc strip (eta-only) measurement
bool isPrecisionHit(const MuonSpacePoint& sp);

/// @brief Returns the identifier of the chamber volume enclosing the surface
///        on which the space point was recorded
constexpr Acts::GeometryIdentifier toChamberId(
    const Acts::GeometryIdentifier& id) {
  return Acts::GeometryIdentifier{}.withVolume(id.volume()).withLayer(id.layer());
}

/// @brief Returns the transform from the space point frame to the global frame.
///        The space points are expressed in the chamber volume frame rotated
///        by 90 degrees around the z-axis (c.f. MuonSpacePointDigitizer)
/// @param volumeLocalToGlobal: Local to global transform of the chamber volume
Acts::Transform3 spacePointFrameToGlobal(
    const Acts::Transform3& volumeLocalToGlobal);

/// @brief Returns the transform from the space point frame to the global frame
///        by looking up the chamber volume in the tracking geometry
/// @param gctx: Geometry context
/// @param trackingGeometry: Tracking geometry holding the chamber volumes
/// @param geoId: Geometry identifier of the space point
Acts::Transform3 spacePointFrameToGlobal(
    const Acts::GeometryContext& gctx,
    const Acts::TrackingGeometry& trackingGeometry,
    const Acts::GeometryIdentifier& geoId);

/// @brief Assigns a logical measurement layer number to each space point in
///        the bucket. The layers are the distinct planes along the local z-axis
///        of the chamber frame, numbered in increasing z starting from 0.
/// @param bucket: Space point bucket of a single chamber
/// @param tolerance: Maximum z-distance of two space points on the same layer
/// @return Layer numbers aligned with the bucket indices
std::vector<std::uint8_t> layerNumbers(const MuonSpacePointBucket& bucket,
                                       double tolerance = s_layerTolerance);

}  // namespace ActsExamples::MuonGlobalPatternFinderUtils
