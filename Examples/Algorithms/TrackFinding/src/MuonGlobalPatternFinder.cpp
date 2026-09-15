// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFinding/MuonGlobalPatternFinder.hpp"

#include "Acts/Definitions/Common.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "Acts/Utilities/StringHelpers.hpp"
#include "Acts/Utilities/VectorHelpers.hpp"
#include "ActsExamples/TrackFinding/MuonGlobalPatternFinderUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <optional>
#include <stdexcept>
#include <utility>

using namespace Acts::UnitLiterals;
using Acts::VectorHelpers::perp;
using Acts::VectorHelpers::phi;

namespace ActsExamples {

using MuonGlobalPatternFinderDefs::detailed;

MuonGlobalPatternFinder::MuonGlobalPatternFinder(
    Config config, std::unique_ptr<const Acts::Logger> logger)
    : m_cfg{std::move(config)}, m_logger{std::move(logger)} {
  if (!m_cfg.localToGlobal) {
    throw std::invalid_argument(
        "MuonGlobalPatternFinder: the local to global transform is not set");
  }
  if (!m_logger) {
    throw std::invalid_argument("MuonGlobalPatternFinder: missing logger");
  }
}

MuonGlobalPatternFinder::SearchTreeData MuonGlobalPatternFinder::constructTree(
    const Acts::GeometryContext& gctx,
    const MuonSpacePointContainer& spacePoints) const {
  using enum MuonExpandedSector::SectorProjector;

  std::vector<HitPayload> hitPayloads{};
  // Reserve the full capacity upfront, as the tree refers to the payloads by
  // pointer
  std::size_t totalHits{0u};
  for (const MuonSpacePointBucket& bucket : spacePoints) {
    totalHits += bucket.size();
  }
  hitPayloads.reserve(totalHits);

  for (const MuonSpacePointBucket& bucket : spacePoints) {
    if (bucket.empty()) {
      continue;
    }
    const MuonStationIndex station =
        stationIndex(bucket.front().id().msStation());
    if (station == MuonStationIndex::UnDef) {
      ACTS_WARNING("constructTree() Skip bucket with undefined station "
                   << bucket.front().id());
      continue;
    }
    const Acts::Transform3 localToGlobal = m_cfg.localToGlobal(gctx, bucket);
    const std::vector<std::uint8_t> layers =
        MuonGlobalPatternFinderUtils::layerNumbers(bucket);

    for (std::size_t i = 0u; i < bucket.size(); ++i) {
      const MuonSpacePoint& sp = bucket[i];
      // Ignore phi-only hits and MDT hits if desired
      if (!sp.id().measuresEta() || (!m_cfg.useMdtHits && sp.isStraw())) {
        continue;
      }
      const HitPayload& newHit = hitPayloads.emplace_back(
          &sp, &bucket, localToGlobal, layers[i], station);
      ACTS_VERBOSE("constructTree() Building hit from "
                   << sp << "\nPhiCov: " << newHit.phiCov << ", Pos: "
                   << Acts::toString(newHit.position)
                   << ", SensorDir: " << Acts::toString(newHit.sensorDir));
    }
  }

  SearchTree_t::vector_t treeData{};
  treeData.reserve(3u * hitPayloads.size());
  for (const HitPayload& hit : hitPayloads) {
    const Acts::Vector3& pos = hit.position;
    // The expanded sector of the hit position is only meaningful if it
    // measures phi
    std::optional<MuonExpandedSector> hitExpSector{};
    if (hit.measuresPhi) {
      hitExpSector.emplace(phi(pos));
    }
    // Duplicate the hit in the neighbouring sectors if it is close to the
    // sector border. This ensures that patterns crossing the sector borders
    // can be found.
    for (const auto proj : {leftOverlap, center, rightOverlap}) {
      const MuonExpandedSector expSect{static_cast<unsigned>(hit->id().sector()),
                                       proj};
      if (proj != center && hitExpSector.has_value() &&
          expSect != *hitExpSector) {
        ACTS_VERBOSE("constructTree() Hit with " << *hitExpSector
                                                 << " is not compatible with "
                                                 << expSect);
        continue;
      }
      // Project the hit onto the plane along the sector radial direction. This
      // removes the bias of the hit displacement in the phi direction.
      const Acts::Vector3 planeNormal = expSect.normalDir();
      const double projR =
          hit.measuresPhi ? perp(pos)
                          : perp(Acts::Vector3{
                                pos - pos.dot(planeNormal) * planeNormal});

      SearchTree_t::coordinate_t coords{};
      coords[Acts::toUnderlying(SeedCoords::eTheta)] =
          std::atan2(projR, pos.z());
      coords[Acts::toUnderlying(SeedCoords::eSector)] = expSect.sector();
      ACTS_VERBOSE("constructTree() Add hit: Z: "
                   << pos.z() << ", R: " << perp(pos) << ", ProjR: " << projR
                   << ", Phi: " << phi(pos) / 1._degree
                   << ", SectorPhi: " << expSect.phi() / 1._degree
                   << " to the search tree");
      treeData.emplace_back(coords, &hit);
    }
  }
  ACTS_DEBUG("constructTree() Create a new tree with "
             << treeData.size() << " entries and " << hitPayloads.size()
             << " hits.");
  return SearchTreeData{std::move(hitPayloads),
                        SearchTree_t{std::move(treeData)}};
}

bool MuonGlobalPatternFinder::passPatternCuts(const PatternState& pat) const {
  const auto nGoodStations = std::ranges::count_if(
      pat.hitsPerStation, [this](const std::vector<CandidateHit>& hits) {
        return hits.size() >= m_cfg.minStationLayers;
      });
  if (pat.nTriggerLayers < m_cfg.minTriggerLayers ||
      pat.nPrecisionLayers < m_cfg.minPrecisionLayers || nGoodStations < 2) {
    ACTS_VERBOSE("passPatternCuts() Pattern "
                 << detailed(pat)
                 << "\ndoes not meet minimum layer requirements - reject.");
    return false;
  }
  if (pat.meanNormResidual2 > m_cfg.meanNormRes2Cut) {
    ACTS_VERBOSE("passPatternCuts() Pattern "
                 << detailed(pat)
                 << "\ndoes not meet the mean norm residual2 cut - reject.");
    return false;
  }
  return true;
}

bool MuonGlobalPatternFinder::isBetter(const PatternState& a,
                                       const PatternState& b) {
  const double resA = a.getMeanResidual2();
  const double resB = b.getMeanResidual2();
  const double resDiff = std::abs(resA - resB) / std::max(resA, resB);
  const int nLayerDiff = a.nBendingLayers() - b.nBendingLayers();
  const int nPrecLayDiff = a.nPrecisionLayers - b.nPrecisionLayers;

  // For patterns that differ by 1-2 layers, don't sacrifice fit quality unless
  // the extra layers are genuinely comparable. For a >= 3-layer difference, the
  // multiplicity advantage is strong enough to dominate.
  if ((nLayerDiff == 0 && nPrecLayDiff == 0) ||
      (std::abs(nLayerDiff) < 3 && resDiff > 0.1)) {
    return resA < resB;
  }
  if (nLayerDiff == 0) {
    return nPrecLayDiff > 0;
  }
  return nLayerDiff > 0;
}

MuonGlobalPatternFinder::LayerOrdering
MuonGlobalPatternFinder::checkLayerOrdering(const HitPayload& hit1,
                                            const HitPayload& hit2) {
  using enum LayerOrdering;
  auto getLayerOrdering = [](bool isLayer1Lower) {
    return isLayer1Lower ? eLowerLayer : eHigherLayer;
  };
  if (hit1 == hit2) {
    return eSameLayer;
  }
  // Hits in the same chamber are ordered by their layer number
  if (hit1.bucket == hit2.bucket) {
    if (hit1.locLayer == hit2.locLayer) {
      return eSameLayer;
    }
    return getLayerOrdering(hit1.locLayer < hit2.locLayer);
  }
  const MuonStationIndex st1 = hit1.station;
  const MuonStationIndex st2 = hit2.station;
  // Hits in the same station but different chambers, e.g. in the overlap
  // region of two adjacent sectors
  if (st1 == st2) {
    const double delta =
        isBarrel(st1)
            ? perp(hit1.position) - perp(hit2.position)
            : std::abs(hit1.position.z()) - std::abs(hit2.position.z());
    if (std::abs(delta) <= Acts::s_epsilon) {
      return eSameLayer;
    }
    return getLayerOrdering(delta < 0.);
  }
  const MuonLayerIndex layer1 = layerIndex(st1);
  const MuonLayerIndex layer2 = layerIndex(st2);
  if (layer1 == layer2) {
    // Hits in different stations but same station layer. Expected to happen
    // only for the Inner and Middle layers
    if (layer1 == MuonLayerIndex::Middle) {
      // The barrel hit comes first
      return getLayerOrdering(st1 == MuonStationIndex::BM);
    }
    if (layer1 == MuonLayerIndex::Inner) {
      // In the large sectors BI comes first, while in the small sectors EI
      // comes first. Use the global radius instead.
      return getLayerOrdering(perp(hit1.position) < perp(hit2.position));
    }
    throw std::runtime_error(
        std::format("MuonGlobalPatternFinder: unexpected pattern-compatible "
                    "hits in {} and {}",
                    toString(st1), toString(st2)));
  }
  if (layer1 == MuonLayerIndex::Inner || layer2 == MuonLayerIndex::Inner) {
    // A hit in the Inner layer comes first
    return getLayerOrdering(layer1 == MuonLayerIndex::Inner);
  }
  if (layer1 == MuonLayerIndex::Outer || layer2 == MuonLayerIndex::Outer) {
    // A hit in the Outer layer comes last
    return getLayerOrdering(layer2 == MuonLayerIndex::Outer);
  }
  if (layer1 == MuonLayerIndex::BarrelExtended ||
      layer2 == MuonLayerIndex::BarrelExtended) {
    // A BarrelExtended hit comes before a Middle or Extended one
    return getLayerOrdering(layer1 == MuonLayerIndex::BarrelExtended);
  }
  // One hit in the Extended (EE) layer and the other one in the Middle layer:
  // EE comes before EM but after BM
  if (layer1 == MuonLayerIndex::Extended) {
    return getLayerOrdering(st2 == MuonStationIndex::EM);
  }
  return getLayerOrdering(st1 == MuonStationIndex::BM);
}

}  // namespace ActsExamples
