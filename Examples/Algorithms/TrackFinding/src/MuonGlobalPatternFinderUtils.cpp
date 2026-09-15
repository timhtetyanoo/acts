// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFinding/MuonGlobalPatternFinderUtils.hpp"

#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"

#include <algorithm>
#include <format>
#include <iterator>
#include <limits>
#include <stdexcept>

using namespace Acts::UnitLiterals;

namespace ActsExamples::MuonGlobalPatternFinderUtils {

bool isPrecisionHit(const MuonSpacePoint& sp) {
  using enum MuonSpacePoint::MuonId::TechField;
  switch (sp.id().technology()) {
    case Mdt:
    case Mm:
      return true;
    case sTgc:
      return sp.id().measuresEta() && !sp.id().measuresPhi();
    default:
      return false;
  }
}

Acts::Transform3 spacePointFrameToGlobal(
    const Acts::Transform3& volumeLocalToGlobal) {
  return volumeLocalToGlobal *
         Acts::AngleAxis3{-90._degree, Acts::Vector3::UnitZ()};
}

Acts::Transform3 spacePointFrameToGlobal(
    const Acts::GeometryContext& gctx,
    const Acts::TrackingGeometry& trackingGeometry,
    const Acts::GeometryIdentifier& geoId) {
  const Acts::TrackingVolume* volume =
      trackingGeometry.findVolume(toChamberId(geoId));
  if (volume == nullptr) {
    throw std::runtime_error(std::format(
        "MuonGlobalPatternFinderUtils: no chamber volume found for {}",
        toChamberId(geoId)));
  }
  return spacePointFrameToGlobal(volume->localToGlobalTransform(gctx));
}

std::vector<std::uint8_t> layerNumbers(const MuonSpacePointBucket& bucket,
                                       double tolerance) {
  std::vector<double> sortedZ{};
  sortedZ.reserve(bucket.size());
  std::ranges::transform(
      bucket, std::back_inserter(sortedZ),
      [](const MuonSpacePoint& sp) { return sp.localPosition().z(); });
  std::ranges::sort(sortedZ);

  /// Each plane is represented by the lowest z-value of its space points
  std::vector<double> planes{};
  for (const double z : sortedZ) {
    if (planes.empty() || z - planes.back() > tolerance) {
      planes.push_back(z);
    }
  }
  if (planes.size() > std::numeric_limits<std::uint8_t>::max() + 1u) {
    throw std::out_of_range(
        std::format("MuonGlobalPatternFinderUtils: {} layers exceed the "
                    "maximum number of layers per bucket",
                    planes.size()));
  }

  std::vector<std::uint8_t> layers{};
  layers.reserve(bucket.size());
  std::ranges::transform(
      bucket, std::back_inserter(layers),
      [&planes, tolerance](const MuonSpacePoint& sp) {
        const auto plane = std::ranges::lower_bound(
            planes, sp.localPosition().z() - tolerance);
        return static_cast<std::uint8_t>(std::distance(planes.begin(), plane));
      });
  return layers;
}

}  // namespace ActsExamples::MuonGlobalPatternFinderUtils
