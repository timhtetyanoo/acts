// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFinding/GlobalPatternFinderAlgorithm.hpp"

#include "Acts/Definitions/Units.hpp"
#include "Acts/Utilities/Enumerate.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "Acts/Utilities/StringHelpers.hpp"
#include "Acts/Utilities/VectorHelpers.hpp"

#include <algorithm>
#include <format>
#include <array>
#include <cmath>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace Acts::UnitLiterals;

namespace ActsExamples {

GlobalPatternFinderAlgorithm::GlobalPatternFinderAlgorithm(
    const Config& cfg, std::unique_ptr<const Acts::Logger> logger)
    : IAlgorithm("GlobalPatternFinderAlgorithm", std::move(logger)),
      m_cfg{cfg} {
  if (m_cfg.inSpacePoints.empty()) {
    throw std::invalid_argument(
        "GlobalPatternFinderAlgorithm: Missing space point collection");
  }
  if (m_cfg.outPatterns.empty()) {
    throw std::invalid_argument(
        "GlobalPatternFinderAlgorithm: Missing output pattern collection");
  }
  if (!m_cfg.trackingGeometry) {
    throw std::invalid_argument(
        "GlobalPatternFinderAlgorithm: Missing tracking geometry");
  }
  m_onlyPhiProvider.trackingGeometry = m_cfg.trackingGeometry.get();
  m_inSpacePoints.initialize(m_cfg.inSpacePoints);
  m_outPatterns.initialize(m_cfg.outPatterns);

  SeedSelector::Config selectorCfg{};
  selectorCfg.seedFromMdt = m_cfg.seedFromMdt;
  selectorCfg.thetaSearchWindow = m_cfg.thetaSearchWindow;
  selectorCfg.seedFromInner = m_cfg.seedFromInner;
  m_seedSelector = std::make_unique<SeedSelector>(std::move(selectorCfg));

  GlobalPatternFinder_t::Config patCfg{};
  patCfg.nResidualSigma = m_cfg.nResidualSigma;
  patCfg.lowConfidenceResSigma = m_cfg.lowConfidenceResSigma;
  patCfg.nPhiSigma = m_cfg.nPhiSigma;
  patCfg.minStripEtaLayers = m_cfg.minTriggerLayers;
  patCfg.minPrecisionLayers = m_cfg.minPrecisionLayers;
  patCfg.minPhiLayers = m_cfg.minPhiLayers;
  patCfg.minGroupLayers = m_cfg.minStationLayers;
  patCfg.meanNormRes2Cut = m_cfg.meanNormRes2Cut;
  patCfg.maxSeedAttempts = m_cfg.maxSeedAttempts;
  patCfg.maxMissLayersInGroup = m_cfg.maxMissLayersInStation;
  patCfg.minHitDistance4Line = m_cfg.minHitDistance4Line;

  m_globPatFinder = std::make_unique<GlobalPatternFinder_t>(
      std::move(patCfg), this->logger().clone("GlobalPatternFinder"));

  // Print Configuration
  ACTS_LOG_WITH_LOGGER(
      this->logger(), Acts::Logging::DEBUG,
      "Global Pattern Finder Configuration:\n"
      << " Theta search window [rad]: " << m_cfg.thetaSearchWindow << "\n"
      << " Number of residual standard deviations: " << m_cfg.nResidualSigma
      << "\n"
      << " Low confidence residual sigma [mm]: " << m_cfg.lowConfidenceResSigma
      << "\n"
      << " Max missed layer hits in station: " << m_cfg.maxMissLayersInStation
      << "\n"
      << " Min hit distance for line [mm]: " << m_cfg.minHitDistance4Line
      << "\n"
      << " Number of phi standard deviations: " << m_cfg.nPhiSigma << "\n"
      << " Min trigger layers: " << m_cfg.minTriggerLayers << "\n"
      << " Min precision layers: " << m_cfg.minPrecisionLayers << "\n"
      << " Min phi layers: " << m_cfg.minPhiLayers << "\n"
      << " Min station layers: " << m_cfg.minStationLayers << "\n"
      << " Mean norm residual^2 cut: " << m_cfg.meanNormRes2Cut << "\n"
      << " Seed from inner: " << m_cfg.seedFromInner << "\n"
      << " Use MDT hits: " << m_cfg.useMdtHits << "\n"
      << " Seed from MDT: " << m_cfg.seedFromMdt << "\n"
      << " Max seed attempts: " << m_cfg.maxSeedAttempts << "\n"
      << " Beam spot radius: " << m_cfg.beamSpotRadius << "\n"
      << " Beam spot length: " << m_cfg.beamSpotLength << "\n");
}

ProcessCode GlobalPatternFinderAlgorithm::execute(
    const AlgorithmContext& ctx) const {
  const MuonSpacePointContainer& inSpacePoints{m_inSpacePoints(ctx)};
  ACTS_DEBUG("Reading " << inSpacePoints.size()
                        << " SP buckets from collection: "
                        << m_cfg.inSpacePoints);

  const Acts::GeometryContext& gctx{ctx.recoGeoContext};

  SearchTreeData treeData{constructTree(gctx, inSpacePoints)};

  GlobalPatternFinder_t::BeamspotInfo beamSpot{};
  beamSpot.position = Acts::Vector3::Zero();
  beamSpot.radius = m_cfg.beamSpotRadius;
  beamSpot.length = m_cfg.beamSpotLength;

  MuonGlobalPatternContainer patterns{
      convertToPattern(m_globPatFinder->findPatterns(
          gctx, treeData.tree, *m_seedSelector, m_onlyPhiProvider, beamSpot))};

  ACTS_DEBUG("Written " << patterns.size()
                        << " GlobalPatterns into the event store.");
  if (logger().doPrint(Acts::Logging::DEBUG)) {
    for (const MuonGlobalPattern& pat : patterns) {
      std::ostringstream hitsPerStation{};
      for (const auto& [st, hits] : Acts::enumerate(pat.hitsPerStation)) {
        if (!hits.empty()) {
          hitsPerStation << " " << static_cast<int>(st) << ":" << hits.size();
        }
      }
      ACTS_DEBUG("Pattern in "
                 << ExpandedSector{pat.sector} << ", theta: " << pat.theta
                 << ", phi: " << pat.phi
                 << ", precision/trigger/phi layers: " << pat.nPrecisionLayers
                 << "/" << pat.nTriggerLayers << "/" << pat.nPhiLayers
                 << ", meanNormRes2: " << pat.meanNormResidual2
                 << ", hits per station:" << hitsPerStation.str());
    }
  }
  /** Write out the global patterns. */
  m_outPatterns(ctx, std::move(patterns));
  return ProcessCode::SUCCESS;
}

SearchTreeData GlobalPatternFinderAlgorithm::constructTree(
    const Acts::GeometryContext& gctx,
    const MuonSpacePointContainer& spacepoints) const {
  std::vector<HitPayload> hitPayloads{};
  using enum ExpandedSector::SectorProjector;
  /** First estimate the number of hits */
  std::size_t totalHits = 0;
  for (const MuonSpacePointBucket& bucket : spacepoints) {
    totalHits += bucket.size();
  }
  hitPayloads.reserve(totalHits);

  for (const MuonSpacePointBucket& bucket : spacepoints) {
    if (bucket.empty()) {
      continue;
    }
    const Acts::Transform3 localToGlobal{
        localToGlobalTransform(gctx, *m_cfg.trackingGeometry, bucket)};

    for (const MuonSpacePoint& hit : bucket) {
      // Ignore only-phi hits and MDT hits if desired
      if (!hit.id().measuresEta() || (!m_cfg.useMdtHits && hit.isStraw())) {
        continue;
      }

      const Acts::Surface* surface{
          m_cfg.trackingGeometry->findSurface(hit.geometryId())};
      if (surface == nullptr) {
        throw std::runtime_error(std::format(
            "GlobalPatternFinderAlgorithm: no surface for geometry id {}",
            hit.geometryId().value()));
      }
      hitPayloads.emplace_back(gctx, &hit, &bucket, localToGlobal, surface);

      if (logger().doPrint(Acts::Logging::VERBOSE)) {
        const HitPayload& newHit{hitPayloads.back()};
        std::ostringstream oss{};
        oss << __func__ << "() Building hit from " << hit << std::endl
            << "PhiCov: " << newHit.phiCov
            << ", Pos: " << Acts::toString(newHit.position) << ", SensorDir: "
            << Acts::toString(newHit.globalSensorDirection(gctx));
        if (newHit.spacePoint()->isStraw()) {
          oss << ", discCov: " << newHit.stripAngle;
        } else {
          oss << "orthogonalStrips: " << !newHit.nonOrthogonalStrips;
        }
        ACTS_VERBOSE(oss.str());
      }
    }
  }

  SearchTree_t::vector_t treeData{};
  /** Athena dereferences hitPayloads.front() below, which is undefined for an
   * empty event */
  if (hitPayloads.empty()) {
    return SearchTreeData{std::move(hitPayloads),
                          SearchTree_t{std::move(treeData)}};
  }
  treeData.reserve(3 * hitPayloads.size());

  std::uint16_t msSector = hitPayloads.front().spacePoint()->id().sector();
  const MuonSpacePointBucket* currentBucket = hitPayloads.front().bucket;
  for (const HitPayload& hit : hitPayloads) {
    ACTS_VERBOSE(__func__ << "() Spacepoint: " << *hit.spacePoint());
    const Acts::Vector3& pos{hit.position};
    const ExpandedSector hitExpSector{Acts::VectorHelpers::phi(pos)};
    if (hit.bucket != currentBucket) {
      currentBucket = hit.bucket;
      msSector = hit.spacePoint()->id().sector();
    }

    /** Try to duplicate the hit in the neighboring sectors if it is close to
     * the sector border. This ensures that we can find patterns crossing the
     * sector borders. */
    for (const ExpandedSector::SectorProjector proj :
         {leftOverlap, center, rightOverlap}) {
      /// Check whether the hit belongs to the left or right sector as well
      const ExpandedSector expSect{msSector, proj};
      if (proj != ExpandedSector::SectorProjector::center &&
          hit.spacePoint()->id().measuresPhi() && expSect != hitExpSector) {
        ACTS_VERBOSE("addHitToTree() Hit with "
                     << hitExpSector << " is not compatible with " << expSect);
        continue;
      }

      /* Project the hit onto the plane along the sector radial direction.
       * This allows to remove the bias of hit displacement in phi direction */
      const Acts::Vector3 planeNormal{expSect.normalDir()};
      const double projR{hit.spacePoint()->id().measuresPhi()
                             ? Acts::VectorHelpers::perp(pos)
                             : Acts::VectorHelpers::perp(Acts::Vector3{
                                   pos - pos.dot(planeNormal) * planeNormal})};

      std::array<double, 2> coords{};
      using HitCoords = GlobalPatternFinder_t::HitCoords;
      coords[Acts::toUnderlying(HitCoords::eTheta)] =
          std::atan2(projR, pos.z());
      coords[Acts::toUnderlying(HitCoords::eSector)] = expSect.sector();

      ACTS_VERBOSE("addHitToTree() Add hit: Z: "
                   << pos.z() << ", R: " << Acts::VectorHelpers::perp(pos)
                   << ", ProjR: " << projR
                   << ", Phi: " << Acts::VectorHelpers::phi(pos) / 1_degree
                   << ", SectorPhi: " << expSect.phi() / 1_degree
                   << " and coordinates [" << coords[0] << ", " << coords[1]
                   << "] to search tree");
      treeData.emplace_back(std::move(coords), &hit);
    }
  }
  ACTS_VERBOSE(__func__ << "() Create a new tree with " << treeData.size()
                        << " entries and " << hitPayloads.size() << " hits.");
  return SearchTreeData{std::move(hitPayloads),
                        SearchTree_t{std::move(treeData)}};
}

MuonGlobalPattern GlobalPatternFinderAlgorithm::convertToPattern(
    const PatternResult& candidate) const {
  MuonGlobalPattern pattern{};
  /** Add eta hits */
  for (PatternTopology::GroupIdx g = 0u; g < PatternTopology::nGroups; ++g) {
    const auto& hits{candidate.hitsPerGroup[g]};
    if (hits.empty())
      continue;

    auto& outHits{pattern.hitsPerStation[g]};
    outHits.reserve(hits.size());

    std::ranges::for_each(hits, [&outHits](const HitPayload* h) {
      outHits.push_back(h->spacePoint());
    });
  }

  /** Add phi-only hits */
  for (const HitPayload& hit : candidate.phiOnlyHits) {
    pattern.hitsPerStation[Acts::toUnderlying(hit.station)].push_back(
        hit.spacePoint());
  }
  pattern.theta = candidate.patTheta;
  pattern.phi = candidate.patPhi;
  // Set the pattern sector(s) and theta.
  pattern.sector = candidate.expSect.sector();
  // Set pattern quality information.
  pattern.nPrecisionLayers = candidate.nPrecisionLayers;
  pattern.nTriggerLayers = candidate.nTriggerLayers;
  pattern.nPhiLayers = candidate.nPhiLayers;
  pattern.meanNormResidual2 = candidate.meanNormResidual2;
  return pattern;
}
MuonGlobalPatternContainer GlobalPatternFinderAlgorithm::convertToPattern(
    const std::vector<PatternResult>& candidates) const {
  MuonGlobalPatternContainer patterns{};
  patterns.reserve(candidates.size());
  std::transform(candidates.begin(), candidates.end(),
                 std::back_inserter(patterns),
                 [this](const PatternResult& candidate) {
                   return convertToPattern(candidate);
                 });
  return patterns;
}

}  // namespace ActsExamples
