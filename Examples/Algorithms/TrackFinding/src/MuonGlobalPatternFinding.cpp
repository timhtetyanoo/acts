// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFinding/MuonGlobalPatternFinding.hpp"

#include "Acts/Geometry/TrackingGeometry.hpp"
#include "ActsExamples/TrackFinding/MuonGlobalPatternFinderUtils.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ActsExamples {

MuonGlobalPatternFinding::MuonGlobalPatternFinding(
    const Config& cfg, std::unique_ptr<const Acts::Logger> logger)
    : IAlgorithm("MuonGlobalPatternFinding", std::move(logger)), m_cfg{cfg} {
  if (m_cfg.inputSpacePoints.empty()) {
    throw std::invalid_argument(
        "MuonGlobalPatternFinding: missing input space point collection");
  }
  if (m_cfg.outputPatterns.empty()) {
    throw std::invalid_argument(
        "MuonGlobalPatternFinding: missing output pattern collection");
  }
  if (!m_cfg.localToGlobal && !m_cfg.trackingGeometry) {
    throw std::invalid_argument(
        "MuonGlobalPatternFinding: either the tracking geometry or the local "
        "to global transform must be provided");
  }

  MuonGlobalPatternFinder::Config finderCfg = m_cfg;
  if (m_cfg.seedFromInner &&
      std::ranges::find(finderCfg.layerSeedings, MuonLayerIndex::Inner) ==
          finderCfg.layerSeedings.end()) {
    finderCfg.layerSeedings.push_back(MuonLayerIndex::Inner);
  }
  if (!finderCfg.localToGlobal) {
    finderCfg.localToGlobal =
        [trackingGeometry = m_cfg.trackingGeometry](
            const Acts::GeometryContext& gctx,
            const MuonSpacePointBucket& bucket) {
          return MuonGlobalPatternFinderUtils::spacePointFrameToGlobal(
              gctx, *trackingGeometry, bucket.front().geometryId());
        };
  }
  m_finder = std::make_unique<const MuonGlobalPatternFinder>(
      std::move(finderCfg), logger().clone("MuonGlobalPatternFinder"));

  m_inputSpacePoints.initialize(m_cfg.inputSpacePoints);
  m_outputPatterns.initialize(m_cfg.outputPatterns);

  ACTS_DEBUG("Global pattern finder configuration:"
             << "\n Theta search window [rad]: " << m_cfg.thetaSearchWindow
             << "\n Number of residual standard deviations: "
             << m_cfg.nResidualSigma
             << "\n Low confidence residual sigma [mm]: "
             << m_cfg.lowConfidenceResSigma
             << "\n Max missed layer hits in station: "
             << m_cfg.maxMissLayersInStation
             << "\n Min hit distance for line [mm]: "
             << m_cfg.minHitDistance4Line
             << "\n Number of phi standard deviations: " << m_cfg.nPhiSigma
             << "\n Min trigger layers: " << m_cfg.minTriggerLayers
             << "\n Min precision layers: " << m_cfg.minPrecisionLayers
             << "\n Min phi layers: " << m_cfg.minPhiLayers
             << "\n Min station layers: " << m_cfg.minStationLayers
             << "\n Mean norm residual^2 cut: " << m_cfg.meanNormRes2Cut
             << "\n Seed from inner: " << m_cfg.seedFromInner
             << "\n Use MDT hits: " << m_cfg.useMdtHits
             << "\n Seed from MDT: " << m_cfg.seedFromMdt
             << "\n Max seed attempts: " << m_cfg.maxSeedAttempts
             << "\n Beam spot radius: " << m_cfg.beamSpotRadius
             << "\n Beam spot length: " << m_cfg.beamSpotLength);
}

MuonGlobalPatternFinding::~MuonGlobalPatternFinding() = default;

ProcessCode MuonGlobalPatternFinding::execute(
    const AlgorithmContext& ctx) const {
  const MuonSpacePointContainer& spacePoints = m_inputSpacePoints(ctx);
  ACTS_DEBUG("Reading " << spacePoints.size() << " space point buckets.");

  MuonGlobalPatternContainer patterns =
      m_finder->findPatterns(ctx.recoGeoContext, spacePoints);

  ACTS_DEBUG("Found " << patterns.size() << " global patterns.");
  if (logger().doPrint(Acts::Logging::VERBOSE)) {
    for (const MuonGlobalPattern& pattern : patterns) {
      ACTS_VERBOSE(pattern);
    }
  }
  m_outputPatterns(ctx, std::move(patterns));
  return ProcessCode::SUCCESS;
}

}  // namespace ActsExamples
