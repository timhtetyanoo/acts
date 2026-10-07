// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFinding/CudaPatternFinder.hpp"

#include "Acts/Definitions/Units.hpp"
#include "Acts/Seeding/detail/CompSpacePointAuxiliaries.hpp"
#include "Acts/Surfaces/detail/LineHelper.hpp"
#include "Acts/Utilities/Enumerate.hpp"
#include "Acts/Utilities/VectorHelpers.hpp"
#include "Acts/Utilities/detail/periodic.hpp"
#include "ActsExamples/Utilities/CudaStream.hpp"
#include "ActsExamples/Utilities/CudaUtilities.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <numbers>
#include <optional>
#include <stdexcept>

namespace ActsExamples {

CudaGlobalPatternFinder::CudaGlobalPatternFinder(
    Config cfg, std::unique_ptr<const Acts::Logger> logger)
    : m_cfg{std::move(cfg)}, m_logger{std::move(logger)} {}

std::vector<CudaGlobalPatternFinder::OutputPattern>
CudaGlobalPatternFinder::findPatterns(
    const Acts::GeometryContext& gctx, const SearchTreeData& treeData,
    const SeedSelector& seedSelector,
    const OnlyPhiHitsProvider& onlyPhiProvider, const BeamspotInfo& beamspot,
    CudaPatternFinderTiming* timing) const {
  using Clock = std::chrono::steady_clock;
  auto microseconds = [](Clock::duration d) {
    return std::chrono::duration<double, std::micro>(d).count();
  };
  const auto fillStart = Clock::now();
  const std::vector<HitPayload>& payloads{treeData.stripPayloads};
  CudaCandidateListContainer lists{
      makeCudaCandidateLists<GlobalPatternFinder_t>(treeData.tree, payloads,
                                                    seedSelector)};
  if (timing != nullptr) {
    timing->nSeeds = lists.nSeeds();
    timing->nCandidates = lists.nCandidates();
  }
  if (lists.empty()) {
    return {};
  }
  CudaHitPayloadContainer hits{makeCudaHitPayloadContainer(gctx, payloads)};
  if (timing != nullptr) {
    timing->fillUs = microseconds(Clock::now() - fillStart);
  }

  CudaStream stream;
  std::array<cudaEvent_t, 2> uploadMarks{};
  if (timing != nullptr) {
    for (cudaEvent_t& mark : uploadMarks) {
      ACTS_CUDA_CHECK(cudaEventCreate(&mark));
    }
    ACTS_CUDA_CHECK(cudaEventRecord(uploadMarks[0], stream.get()));
  }
  hits.moveToDevice(stream.get());
  lists.moveToDevice(stream.get());
  if (timing != nullptr) {
    ACTS_CUDA_CHECK(cudaEventRecord(uploadMarks[1], stream.get()));
  }
  const CudaPatternRecordContainer event{findCudaPatternsInEta(
      hits, lists, makeCudaPatternFinderConfig(m_cfg.finder, beamspot),
      m_cfg.sizes, stream.get(), timing)};
  if (timing != nullptr) {
    float ms{0.f};
    ACTS_CUDA_CHECK(cudaEventElapsedTime(&ms, uploadMarks[0], uploadMarks[1]));
    timing->uploadUs = 1000. * ms;
    for (cudaEvent_t& mark : uploadMarks) {
      ACTS_CUDA_CHECK(cudaEventDestroy(mark));
    }
  }
  if (const std::uint8_t overflow{event.overflow(0u)}; overflow != 0u) {
    ACTS_WARNING(__func__ << "() Patterns did not fit the device buffers "
                          << "(CudaPatternOverflow bits "
                          << static_cast<int>(overflow)
                          << "); the seeds that did not fit have no patterns");
  }
  const auto tailStart = Clock::now();

  std::vector<PatternState> patterns{};
  for (std::uint32_t k = 0u; k < event.nPatterns(0u); ++k) {
    const std::size_t slot{event.slot(0u, k)};
    if (event.pattern(slot).status == CudaPatternStatus::ok) {
      patterns.push_back(makePatternState(gctx, payloads, event, slot));
    }
  }
  ACTS_DEBUG(__func__ << "() Found " << patterns.size()
                      << " patterns in eta from " << lists.nSeeds()
                      << " seeds");

  addPhiOnlyHits(gctx, onlyPhiProvider, patterns);

  // GlobalPatternFinder::convertToPattern
  std::vector<OutputPattern> output{};
  output.reserve(patterns.size());
  for (PatternState& pat : patterns) {
    OutputPattern& out{output.emplace_back(pat.expSect.sector())};
    for (std::size_t g = 0u; g < PatternTopology::nGroups; ++g) {
      for (const auto& hit : pat.hitsPerGroup[g]) {
        out.hitsPerGroup[g].push_back(hit.hitPtr);
      }
    }
    out.phiOnlyHits = std::move(pat.phiOnlyHits);
    out.meanNormResidual2 = pat.meanNormResidual2;
    out.patTheta = pat.patTheta;
    out.patPhi = pat.patPhi;
    out.nPrecisionLayers = static_cast<std::uint8_t>(pat.nPrecisionLayers);
    out.nTriggerLayers = static_cast<std::uint8_t>(pat.nTriggerLayers);
    out.nPhiLayers = static_cast<std::uint8_t>(pat.nPhiLayers);
  }
  if (timing != nullptr) {
    timing->tailUs = microseconds(Clock::now() - tailStart);
  }
  return output;
}

CudaGlobalPatternFinder::PatternState CudaGlobalPatternFinder::makePatternState(
    const Acts::GeometryContext& gctx, const std::vector<HitPayload>& payloads,
    const CudaPatternRecordContainer& event, std::size_t slot) const {
  using OrderedHit = PatternState::OrderedHit;
  const CudaPatternRecordRow row{event.pattern(slot)};
  auto orderedHit = [&](std::size_t i) {
    return OrderedHit{&payloads[event.hitIndex(slot, i)],
                      event.globLayer(slot, i)};
  };
  PatternState pat{gctx, orderedHit(0u), row.sector, &m_cfg.finder,
                   m_logger.get()};
  pat.hitsPerGroup = {};
  for (std::size_t i = 0u; i < row.nHits; ++i) {
    const OrderedHit hit{orderedHit(i)};
    pat.hitsPerGroup[PatternTopology::groupIndex(*hit)].push_back(hit);
  }
  pat.lastInsertedHit = orderedHit(row.lastInsertedHit);
  pat.prevLayerHit = row.prevLayerHit != cudaInvalidHitPosition
                         ? orderedHit(row.prevLayerHit)
                         : pat.lastInsertedHit;
  // The device keeps an anchor that a branch replaced; it is not in the hits
  pat.lineAnchorHit = row.lineAnchorHit != cudaInvalidHitPosition
                          ? orderedHit(row.lineAnchorHit)
                          : pat.lastInsertedHit;
  pat.bendPlaneNorm = row.bendPlaneNorm;
  pat.linePos = row.linePos;
  pat.lineDir = row.lineDir;
  pat.leverArm = row.leverArm;
  pat.meanNormResidual2 = row.meanNormResidual2;
  pat.lastResidual = row.lastResidual;
  pat.lastResSigma = row.lastResSigma;
  pat.patPhi = row.patPhi;
  pat.patTheta = row.patTheta;
  pat.patPhiCov = row.patPhiCov;
  pat.nPrecisionLayers = row.nPrecisionLayers;
  pat.nTriggerLayers = row.nTriggerLayers;
  pat.nPhiLayers = row.nPhiLayers;
  pat.isFinalized = true;
  pat.useBeamspot = (row.flags & CudaPatternFlags::useBeamspot) != 0u;
  pat.needLineUpdate = (row.flags & CudaPatternFlags::needLineUpdate) != 0u;
  return pat;
}

void CudaGlobalPatternFinder::addPhiOnlyHits(
    const Acts::GeometryContext& gctx,
    const OnlyPhiHitsProvider& onlyPhiProvider,
    std::vector<PatternState>& patterns) const {
  using OrderedHit = PatternState::OrderedHit;
  // The pattern line in the group of a phi hit, from its eta hits or from
  // the closest eta hits around it
  auto computePatternLineInGroup = [&](PatternState& pat,
                                       const HitPayload& phiHit) -> bool {
    const auto phiHitGroup{PatternTopology::groupIndex(phiHit)};
    const auto& groupHits{pat.hitsPerGroup[phiHitGroup]};

    // useBeamspot flags whether the line could be drawn from the hits
    pat.useBeamspot = true;

    if (groupHits.size() > 1) {
      const auto [minIt, maxIt]{std::ranges::minmax_element(
          groupHits, {}, [](const OrderedHit& c) { return c.globLayer; })};
      pat.lineAnchorHit = *minIt;
      pat.lastInsertedHit = *maxIt;
      pat.updateLineParameters(gctx, BeamspotInfo{});
    }
    if (pat.useBeamspot && !groupHits.empty()) {
      pat.moveLineAnchorHit(groupHits.front());
      pat.lastInsertedHit =
          *std::ranges::max_element(groupHits, {}, [&](const OrderedHit& c) {
            return (pat.projToPhiPlane(gctx, *c) -
                    pat.projToPhiPlane(gctx, *pat.lineAnchorHit))
                .norm();
          });
      pat.updateLineParameters(gctx, BeamspotInfo{});
    }
    if (pat.useBeamspot) {
      const OrderedHit* lowerHit{nullptr};
      const OrderedHit* upperHit{nullptr};
      for (const auto& [group, hits] : Acts::enumerate(pat.hitsPerGroup)) {
        if (hits.empty() || group == phiHitGroup) {
          continue;
        }
        if (PatternTopology::layerSorter(*hits.front(), phiHit)) {
          const OrderedHit& lowerInGroup{*std::ranges::max_element(
              hits, {}, [](const OrderedHit& c) { return c.globLayer; })};
          if (!lowerHit || lowerInGroup.globLayer > lowerHit->globLayer) {
            lowerHit = &lowerInGroup;
          }
        } else {
          const OrderedHit& upperInGroup{*std::ranges::min_element(
              hits, {}, [](const OrderedHit& c) { return c.globLayer; })};
          if (!upperHit || upperInGroup.globLayer < upperHit->globLayer) {
            upperHit = &upperInGroup;
          }
        }
      }
      if (!lowerHit || !upperHit) {
        return false;
      }
      pat.lineAnchorHit = *lowerHit;
      pat.lastInsertedHit = *upperHit;
      pat.updateLineParameters(gctx, BeamspotInfo{});
    }
    return !pat.useBeamspot;
  };

  std::vector<PatternState> survivingPatterns{};
  survivingPatterns.reserve(patterns.size());
  for (PatternState& pat : patterns) {
    std::optional<PatternTopology::GroupIdx> patternLineGroup{std::nullopt};

    auto projOntoPhiPlane = [&pat](const Acts::Vector3& pos) -> Acts::Vector3 {
      return pos - pos.dot(pat.bendPlaneNorm) * pat.bendPlaneNorm;
    };
    auto phiPull = [&](const HitPayload& hit) {
      return std::abs(Acts::detail::difference_periodic(
                 Acts::VectorHelpers::phi(hit.globalPosition(gctx)), pat.patPhi,
                 2. * std::numbers::pi)) /
             std::sqrt(hit.phiVariance(gctx));
    };

    auto phiHitsPerGroup{onlyPhiProvider.getPhiOnlyHits(pat, gctx)};
    for (auto [group, bucket] : Acts::enumerate(phiHitsPerGroup)) {
      for (HitPayload& newHit : bucket) {
        // Reject hits from a layer that already has an eta hit measuring phi
        const auto& groupHits{pat.hitsPerGroup[group]};
        if (std::ranges::any_of(groupHits, [&](const OrderedHit& h) {
              return h->spacePoint()->measuresLoc0() &&
                     PatternTopology::sameLayer(*h, newHit);
            })) {
          continue;
        }
        if (!pat.isPhiCompatible(gctx, newHit)) {
          continue;
        }
        if (!patternLineGroup || *patternLineGroup != group) {
          if (!computePatternLineInGroup(pat, newHit)) {
            continue;
          }
          patternLineGroup = static_cast<PatternTopology::GroupIdx>(group);
        }
        // The pattern line has to cross the strip of the phi hit
        const Acts::Vector3 newHitPos{newHit.globalPosition(gctx)};
        using SeedingAux =
            Acts::Experimental::detail::CompSpacePointAuxiliaries;
        const double stripHalfLength{
            std::sqrt(newHit.spacePoint()->covariance()[Acts::toUnderlying(
                SeedingAux::ResidualIdx::bending)])};
        const Acts::Vector3 sensorDir{newHit.globalSensorDirection(gctx)};
        const Acts::Vector3 stripLow{
            projOntoPhiPlane(newHitPos - stripHalfLength * sensorDir)};
        const Acts::Vector3 stripHigh{
            projOntoPhiPlane(newHitPos + stripHalfLength * sensorDir)};
        const Acts::Vector3 stripDirOnPlane{
            (stripHigh - stripLow).normalized()};
        const double stripIntersect{
            Acts::detail::LineHelper::lineIntersect<3>(
                pat.linePos, pat.lineDir, stripLow, stripDirOnPlane)
                .pathLength()};
        const double stripProjLength{(stripHigh - stripLow).norm()};

        using namespace Acts::UnitLiterals;
        constexpr double margin{10._mm};
        if (stripIntersect < -margin ||
            stripIntersect > (stripProjLength + margin)) {
          continue;
        }
        // One phi hit per layer, the one closest to the pattern phi
        if (const auto it = std::ranges::find_if(
                pat.phiOnlyHits,
                [&](const HitPayload& h) {
                  return PatternTopology::sameLayer(h, newHit);
                });
            it != pat.phiOnlyHits.end()) {
          if (phiPull(*it) < phiPull(newHit)) {
            continue;
          }
          *it = std::move(newHit);
        } else {
          pat.phiOnlyHits.push_back(std::move(newHit));
          pat.nPhiLayers++;
        }
        pat.updatePatternPhi(gctx);
      }
    }
    if (pat.nPhiLayers < m_cfg.finder.minPhiLayers) {
      continue;
    }
    survivingPatterns.push_back(std::move(pat));
  }
  std::swap(patterns, survivingPatterns);
}

CudaGlobalPatternFinderAlgorithm::CudaGlobalPatternFinderAlgorithm(
    const Config& cfg, std::unique_ptr<const Acts::Logger> logger)
    : IAlgorithm("CudaGlobalPatternFinderAlgorithm", std::move(logger)),
      m_cfg{cfg} {
  if (m_cfg.inSpacePoints.empty()) {
    throw std::invalid_argument(
        "CudaGlobalPatternFinderAlgorithm: Missing space point collection");
  }
  if (m_cfg.outPatterns.empty()) {
    throw std::invalid_argument(
        "CudaGlobalPatternFinderAlgorithm: Missing output pattern collection");
  }
  if (!m_cfg.trackingGeometry) {
    throw std::invalid_argument(
        "CudaGlobalPatternFinderAlgorithm: Missing tracking geometry");
  }
  m_onlyPhiProvider.trackingGeometry = m_cfg.trackingGeometry.get();
  m_inSpacePoints.initialize(m_cfg.inSpacePoints);
  m_outPatterns.initialize(m_cfg.outPatterns);
  if (!m_cfg.outTiming.empty()) {
    m_outTiming.initialize(m_cfg.outTiming);
  }

  SeedSelector::Config selectorCfg{};
  selectorCfg.seedFromMdt = m_cfg.seedFromMdt;
  selectorCfg.thetaSearchWindow = m_cfg.thetaSearchWindow;
  selectorCfg.seedFromInner = m_cfg.seedFromInner;
  m_seedSelector = std::make_unique<SeedSelector>(std::move(selectorCfg));

  // The mapping of GlobalPatternFinderAlgorithm
  CudaGlobalPatternFinder::Config finderCfg{};
  GlobalPatternFinder_t::Config& patCfg{finderCfg.finder};
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
  finderCfg.sizes =
      CudaPatternFinderSizes{.batchSize = m_cfg.batchSize,
                             .workspaceBytes = m_cfg.workspaceBytes,
                             .maxHitsPerPattern = m_cfg.maxHitsPerPattern,
                             .maxPatterns = m_cfg.maxPatterns};
  m_finder = std::make_unique<CudaGlobalPatternFinder>(
      std::move(finderCfg), this->logger().clone("CudaGlobalPatternFinder"));
}

ProcessCode CudaGlobalPatternFinderAlgorithm::execute(
    const AlgorithmContext& ctx) const {
  using Clock = std::chrono::steady_clock;
  const MuonSpacePointContainer& inSpacePoints{m_inSpacePoints(ctx)};
  const Acts::GeometryContext& gctx{ctx.recoGeoContext};

  CudaPatternFinderTiming timing{};
  const auto treeStart = Clock::now();
  const SearchTreeData treeData{constructTree(gctx, inSpacePoints)};
  timing.treeUs =
      std::chrono::duration<double, std::micro>(Clock::now() - treeStart)
          .count();

  GlobalPatternFinder_t::BeamspotInfo beamSpot{};
  beamSpot.position = Acts::Vector3::Zero();
  beamSpot.radius = m_cfg.beamSpotRadius;
  beamSpot.length = m_cfg.beamSpotLength;

  const auto found{m_finder->findPatterns(
      gctx, treeData, *m_seedSelector, m_onlyPhiProvider, beamSpot, &timing)};

  // GlobalPatternFinderAlgorithm::convertToPattern
  MuonGlobalPatternContainer patterns{};
  patterns.reserve(found.size());
  for (const auto& candidate : found) {
    MuonGlobalPattern& pattern{patterns.emplace_back()};
    for (std::size_t g = 0u; g < PatternTopology::nGroups; ++g) {
      for (const HitPayload* hit : candidate.hitsPerGroup[g]) {
        pattern.hitsPerStation[g].push_back(hit->spacePoint());
      }
    }
    for (const HitPayload& hit : candidate.phiOnlyHits) {
      pattern.hitsPerStation[Acts::toUnderlying(hit.station)].push_back(
          hit.spacePoint());
    }
    pattern.theta = candidate.patTheta;
    pattern.phi = candidate.patPhi;
    pattern.sector = candidate.expSect.sector();
    pattern.nPrecisionLayers = candidate.nPrecisionLayers;
    pattern.nTriggerLayers = candidate.nTriggerLayers;
    pattern.nPhiLayers = candidate.nPhiLayers;
    pattern.meanNormResidual2 = candidate.meanNormResidual2;
  }
  ACTS_DEBUG("Found " << patterns.size() << " patterns from " << timing.nSeeds
                      << " seeds");

  m_outPatterns(ctx, std::move(patterns));
  if (m_outTiming.isInitialized()) {
    m_outTiming(ctx, std::move(timing));
  }
  return ProcessCode::SUCCESS;
}

SearchTreeData CudaGlobalPatternFinderAlgorithm::constructTree(
    const Acts::GeometryContext& gctx,
    const MuonSpacePointContainer& spacePoints) const {
  using enum ExpandedSector::SectorProjector;
  std::vector<HitPayload> hitPayloads{};
  std::size_t totalHits{0u};
  for (const MuonSpacePointBucket& bucket : spacePoints) {
    totalHits += bucket.size();
  }
  hitPayloads.reserve(totalHits);

  for (const MuonSpacePointBucket& bucket : spacePoints) {
    if (bucket.empty()) {
      continue;
    }
    const Acts::Transform3 localToGlobal{
        bucketToGlobalTransform(gctx, *m_cfg.trackingGeometry, bucket)};
    for (const MuonSpacePoint& hit : bucket) {
      // Only-phi hits are not in the tree, and MDT hits only if asked for
      if (!hit.id().measuresEta() || (!m_cfg.useMdtHits && hit.isStraw())) {
        continue;
      }
      hitPayloads.emplace_back(
          gctx, &hit, &bucket, localToGlobal,
          &measurementSurface(*m_cfg.trackingGeometry, hit));
    }
  }

  SearchTree_t::vector_t treeData{};
  if (hitPayloads.empty()) {
    return SearchTreeData{std::move(hitPayloads),
                          SearchTree_t{std::move(treeData)}};
  }
  treeData.reserve(3 * hitPayloads.size());

  std::uint16_t msSector = hitPayloads.front().spacePoint()->id().sector();
  const MuonSpacePointBucket* currentBucket = hitPayloads.front().bucket;
  for (const HitPayload& hit : hitPayloads) {
    const Acts::Vector3& pos{hit.position};
    const ExpandedSector hitExpSector{Acts::VectorHelpers::phi(pos)};
    if (hit.bucket != currentBucket) {
      currentBucket = hit.bucket;
      msSector = hit.spacePoint()->id().sector();
    }
    // A hit close to a sector border also enters the neighbouring sector
    for (const ExpandedSector::SectorProjector proj :
         {leftOverlap, center, rightOverlap}) {
      const ExpandedSector expSect{msSector, proj};
      if (proj != center && hit.spacePoint()->id().measuresPhi() &&
          expSect != hitExpSector) {
        continue;
      }
      // Projected onto the plane along the sector radial direction
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
      treeData.emplace_back(std::move(coords), &hit);
    }
  }
  return SearchTreeData{std::move(hitPayloads),
                        SearchTree_t{std::move(treeData)}};
}

}  // namespace ActsExamples
