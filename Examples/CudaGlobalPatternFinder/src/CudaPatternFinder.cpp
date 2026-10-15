// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFinding/CudaPatternFinder.hpp"

#include "Acts/Definitions/Units.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Seeding/detail/CompSpacePointAuxiliaries.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Surfaces/detail/LineHelper.hpp"
#include "Acts/Utilities/Enumerate.hpp"
#include "Acts/Utilities/KDTree.hpp"
#include "Acts/Utilities/VectorHelpers.hpp"
#include "Acts/Utilities/detail/periodic.hpp"
#include "ActsExamples/Utilities/CudaStream.hpp"
#include "ActsExamples/Utilities/CudaUtilities.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <format>
#include <numbers>
#include <optional>
#include <stdexcept>

namespace ActsExamples {

namespace {

/// The search tree over the rows of the hits the finding uses
using RowTree = Acts::KDTree<2, std::uint32_t, double, std::array, 50>;

/// SeedSelector::goodForSeeding and thetaSearchWindow of a space point: the
/// window, or nothing if the hit does not seed
std::optional<double> seedWindow(const SeedSelector& selector,
                                 const MuonSpacePoint& sp) {
  const LayerIndex layer{toLayerIndex(toStationIndex(sp.id().msStation()))};
  if (std::ranges::find(selector.m_seedinglayers, layer) ==
          selector.m_seedinglayers.end() ||
      (sp.isStraw() && !selector.m_cfg.seedFromMdt)) {
    return std::nullopt;
  }
  return (layer == LayerIndex::Inner || layer == LayerIndex::Outer)
             ? selector.m_cfg.thetaSearchWindow
             : 0.5 * selector.m_cfg.thetaSearchWindow;
}

/// Writes a rotation column major
void storeRotation(const Acts::RotationMatrix3& rotation, double* out) {
  for (int c = 0; c < 3; ++c) {
    for (int r = 0; r < 3; ++r) {
      out[3 * c + r] = rotation(r, c);
    }
  }
}

}  // namespace

/// What one event needs on the device side, kept between events: the stream,
/// page-locked memory for the copies up and back, and the device block of
/// the inputs. Events running at the same time take different ones
struct CudaGlobalPatternFinder::Workspace {
  CudaStream stream{};
  CudaPinnedBuffer upload{};
  CudaDeviceBuffer device{};
  CudaPinnedBuffer readback{};
};

std::unique_ptr<CudaGlobalPatternFinder::Workspace>
CudaGlobalPatternFinder::acquireWorkspace() const {
  {
    const std::lock_guard lock{m_workspaceMutex};
    if (!m_workspaces.empty()) {
      std::unique_ptr<Workspace> workspace{std::move(m_workspaces.back())};
      m_workspaces.pop_back();
      return workspace;
    }
  }
  return std::make_unique<Workspace>();
}

void CudaGlobalPatternFinder::releaseWorkspace(
    std::unique_ptr<Workspace> workspace) const {
  if (workspace == nullptr) {
    return;
  }
  const std::lock_guard lock{m_workspaceMutex};
  m_workspaces.push_back(std::move(workspace));
}

CudaGlobalPatternFinder::WorkspaceReturn::~WorkspaceReturn() {
  finder.releaseWorkspace(std::move(workspace));
}

CudaGlobalPatternFinder::~CudaGlobalPatternFinder() {
  if (m_surfaces.geometryId != nullptr) {
    (void)cudaFree(m_surfaces.geometryId);
  }
  if (m_surfaces.rotation != nullptr) {
    (void)cudaFree(m_surfaces.rotation);
  }
}

const CudaGlobalPatternFinder::SurfaceTable&
CudaGlobalPatternFinder::surfaceTable(const Acts::GeometryContext& gctx) const {
  std::call_once(m_surfacesOnce, [&]() {
    if (!m_cfg.trackingGeometry) {
      throw std::invalid_argument(
          "CudaGlobalPatternFinder: the tracking geometry is needed");
    }
    std::vector<std::pair<std::uint64_t, const Acts::Surface*>> surfaces{};
    m_cfg.trackingGeometry->visitSurfaces(
        [&surfaces](const Acts::Surface* surface) {
          surfaces.emplace_back(surface->geometryId().value(), surface);
        },
        false);
    std::ranges::sort(surfaces, {}, [](const auto& s) { return s.first; });
    const auto [first, last] = std::ranges::unique(
        surfaces, {}, [](const auto& s) { return s.first; });
    surfaces.erase(first, last);

    std::vector<std::uint64_t> ids(surfaces.size());
    std::vector<double> rotations(9u * surfaces.size());
    for (std::size_t s = 0u; s < surfaces.size(); ++s) {
      ids[s] = surfaces[s].first;
      storeRotation(surfaces[s].second->localToGlobalTransform(gctx).linear(),
                    rotations.data() + 9u * s);
    }
    m_surfaces.nSurfaces = static_cast<std::uint32_t>(ids.size());
    allocateDeviceColumn(m_surfaces.geometryId, ids.size());
    allocateDeviceColumn(m_surfaces.rotation, rotations.size());
    ACTS_CUDA_CHECK(cudaMemcpy(m_surfaces.geometryId, ids.data(),
                               ids.size() * sizeof(std::uint64_t),
                               cudaMemcpyHostToDevice));
    ACTS_CUDA_CHECK(cudaMemcpy(m_surfaces.rotation, rotations.data(),
                               rotations.size() * sizeof(double),
                               cudaMemcpyHostToDevice));
    ACTS_DEBUG(__func__ << "() " << ids.size()
                        << " measurement surfaces on the device");
    // The memory the pattern building takes from, a share of what the device
    // has free once the table is there
    if (m_cfg.workspaceFraction > 0.) {
      std::size_t free{0u};
      std::size_t total{0u};
      ACTS_CUDA_CHECK(cudaMemGetInfo(&free, &total));
      m_arenaBytes = static_cast<std::size_t>(m_cfg.workspaceFraction *
                                              static_cast<double>(free));
      ACTS_INFO("Pattern building memory: " << m_arenaBytes / (1u << 20u)
                                            << " MiB of " << free / (1u << 20u)
                                            << " MiB free on the device");
    }
  });
  return m_surfaces;
}

MuonGlobalPatternContainer CudaGlobalPatternFinder::findPatterns(
    const Acts::GeometryContext& gctx,
    const MuonSpacePointContainer& spacePoints,
    const SeedSelector& seedSelector,
    const OnlyPhiHitsProvider& onlyPhiProvider, const BeamspotInfo& beamspot,
    CudaPatternFinderTiming* timing) const {
  using Clock = std::chrono::steady_clock;
  auto microseconds = [](Clock::duration d) {
    return std::chrono::duration<double, std::micro>(d).count();
  };
  const auto hostPreStart = Clock::now();
  using enum ExpandedSector::SectorProjector;
  const SurfaceTable& surfaces{surfaceTable(gctx)};
  const Acts::TrackingGeometry& geometry{*m_cfg.trackingGeometry};

  // The global positions of the hits the finding uses and the search tree
  // over them, as GlobalPatternFinderAlgorithm::constructTree
  // Per row: the bucket and the space point, for the payloads of the kept
  // patterns; per bucket: its rows
  std::vector<std::uint32_t> usedBucket{};
  std::vector<double> positionX{};
  std::vector<double> positionY{};
  std::vector<double> positionZ{};
  std::vector<const MuonSpacePoint*> usedPointer{};
  std::vector<double> bucketRotation(9u * spacePoints.size(), 0.);
  std::vector<Acts::Transform3> bucketTransform(spacePoints.size(),
                                                Acts::Transform3::Identity());
  std::vector<std::uint32_t> bucketBegin(spacePoints.size(), 0u);
  std::vector<std::uint32_t> bucketEnd(spacePoints.size(), 0u);
  RowTree::vector_t treeEntries{};
  for (std::size_t k = 0u; k < spacePoints.size(); ++k) {
    const MuonSpacePointBucket& bucket{spacePoints[k]};
    bucketBegin[k] = static_cast<std::uint32_t>(usedPointer.size());
    bucketEnd[k] = bucketBegin[k];
    if (bucket.empty()) {
      continue;
    }
    const Acts::Transform3 localToGlobal{
        bucketToGlobalTransform(gctx, geometry, bucket)};
    bucketTransform[k] = localToGlobal;
    storeRotation(localToGlobal.linear(), bucketRotation.data() + 9u * k);
    // The sector of the bucket, from its first hit
    const unsigned msSector{bucket.front().id().sector()};
    for (const MuonSpacePoint& sp : bucket) {
      // Only-phi hits are not in the tree, and MDT hits only if asked for
      if (!sp.id().measuresEta() || (!m_cfg.useMdtHits && sp.isStraw())) {
        continue;
      }
      const auto row = static_cast<std::uint32_t>(usedPointer.size());
      const Acts::Vector3 pos{localToGlobal * sp.localPosition()};
      usedBucket.push_back(static_cast<std::uint32_t>(k));
      positionX.push_back(pos.x());
      positionY.push_back(pos.y());
      positionZ.push_back(pos.z());
      usedPointer.push_back(&sp);
      // As GlobalPatternFinderAlgorithm::constructTree: a phi-measuring hit
      // enters an overlap sector if its phi is inside it, with its own theta;
      // the others enter all three, projected onto each sector's plane
      const bool measuresPhi{sp.id().measuresPhi()};
      const double r2{pos.head<2>().squaredNorm()};
      for (const ExpandedSector::SectorProjector proj :
           {leftOverlap, center, rightOverlap}) {
        const ExpandedSector expSect{msSector, proj};
        if (measuresPhi && proj != center &&
            !expSect.insideSector(Acts::VectorHelpers::phi(pos))) {
          continue;
        }
        const double theta{
            measuresPhi
                ? Acts::VectorHelpers::theta(pos)
                : std::atan2(
                      std::sqrt(std::max(
                          0., r2 - Acts::square(pos.dot(expSect.normalDir())))),
                      pos.z())};
        std::array<double, 2> coords{};
        using HitCoords = GlobalPatternFinder_t::HitCoords;
        coords[Acts::toUnderlying(HitCoords::eTheta)] = theta;
        coords[Acts::toUnderlying(HitCoords::eSector)] = expSect.sector();
        treeEntries.emplace_back(coords, row);
      }
    }
    bucketEnd[k] = static_cast<std::uint32_t>(usedPointer.size());
  }
  const RowTree tree{std::move(treeEntries)};

  // The candidate lists, as the CPU seed loop's range searches
  using HitCoords = GlobalPatternFinder_t::HitCoords;
  constexpr auto sectorIdx = Acts::toUnderlying(HitCoords::eSector);
  constexpr auto thetaIdx = Acts::toUnderlying(HitCoords::eTheta);
  CudaCandidateListContainer lists{};
  std::vector<std::uint32_t> candidates{};
  for (const auto& [seedCoords, seedRow] : tree) {
    const std::optional<double> window{
        seedWindow(seedSelector, *usedPointer[seedRow])};
    if (!window) {
      continue;
    }
    RowTree::range_t selectRange{};
    selectRange[sectorIdx].shrink(seedCoords[sectorIdx] - 0.1,
                                  seedCoords[sectorIdx] + 0.1);
    selectRange[thetaIdx].shrink(seedCoords[thetaIdx] - *window / 2.,
                                 seedCoords[thetaIdx] + *window / 2.);
    candidates.clear();
    tree.rangeSearchMapDiscard(
        selectRange, [&candidates](const auto& /*coords*/, std::uint32_t row) {
          candidates.push_back(row);
        });
    lists.addSeed(seedRow, seedCoords[sectorIdx], seedCoords[thetaIdx],
                  candidates);
  }
  if (timing != nullptr) {
    timing->nSeeds = lists.nSeeds();
    timing->nCandidates = lists.nCandidates();
  }
  if (lists.empty()) {
    return {};
  }
  // Up to the device in one copy, from page-locked memory kept between
  // events: the columns of the used hits that the payloads read (row r being
  // space point r, every bucket keeping its number), their positions, the
  // bucket rotations and the lists. The payloads, which the device builds,
  // follow the copied part in the same device block
  std::unique_ptr<Workspace> workspace{acquireWorkspace()};
  const WorkspaceReturn giveBack{*this, workspace};
  const cudaStream_t stream{workspace->stream.get()};
  const auto nUsed = static_cast<std::uint32_t>(usedPointer.size());
  const auto nBuckets = static_cast<std::uint32_t>(spacePoints.size());
  CudaMuonSpacePointArrays deviceSpacePoints{};
  CudaUsedHitArrays used{.nHits = nUsed, .nBuckets = nBuckets};
  std::uint32_t* deviceMissing{nullptr};
  CudaHitPayloadContainer hits{};
  std::size_t uploadBytes{0u};
  auto stage = [&](CudaStagedBlock& block) {
    const bool fill{block.host != nullptr};
    auto column = [&](auto*& device, std::size_t n) {
      using T = std::remove_cv_t<
          std::remove_pointer_t<std::remove_reference_t<decltype(device)>>>;
      const auto [host, dev] = block.take<T>(n);
      device = dev;
      return host;
    };
    auto copy = [&](auto*& device, const auto& values) {
      auto* host = column(device, values.size());
      if (fill && host != nullptr) {
        std::ranges::copy(values, host);
      }
    };
    auto* geometryId = column(deviceSpacePoints.geometryId, nUsed);
    auto* muonId = column(deviceSpacePoints.muonId, nUsed);
    auto* localPositionY = column(deviceSpacePoints.localPositionY, nUsed);
    auto* toNextSensorX = column(deviceSpacePoints.toNextSensorX, nUsed);
    auto* toNextSensorY = column(deviceSpacePoints.toNextSensorY, nUsed);
    auto* toNextSensorZ = column(deviceSpacePoints.toNextSensorZ, nUsed);
    auto* covariance0 = column(deviceSpacePoints.covariance0, nUsed);
    auto* covariance1 = column(deviceSpacePoints.covariance1, nUsed);
    auto* driftRadius = column(deviceSpacePoints.driftRadius, nUsed);
    if (fill) {
      for (std::uint32_t row = 0u; row < nUsed; ++row) {
        const MuonSpacePoint& sp{*usedPointer[row]};
        geometryId[row] = sp.geometryId().value();
        muonId[row] = sp.id().toInt();
        localPositionY[row] = sp.localPosition().y();
        toNextSensorX[row] = sp.toNextSensor().x();
        toNextSensorY[row] = sp.toNextSensor().y();
        toNextSensorZ[row] = sp.toNextSensor().z();
        covariance0[row] = sp.covariance()[0];
        covariance1[row] = sp.covariance()[1];
        driftRadius[row] = sp.driftRadius();
      }
    }
    copy(deviceSpacePoints.bucketStart, bucketBegin);
    copy(used.positionX, positionX);
    copy(used.positionY, positionY);
    copy(used.positionZ, positionZ);
    copy(used.bucketRotation, bucketRotation);
    lists.stageForDevice(block);
    // Zero, as the host block holds it
    auto* missing = column(deviceMissing, 1u);
    if (fill) {
      *missing = 0u;
    }
    uploadBytes = block.cursor;
    hits.placeOnDevice(nUsed, block);
  };
  CudaStagedBlock sizing{};
  stage(sizing);
  // The number of hits without a surface comes back after the block
  const std::size_t missingAt{(sizing.cursor + 15u) & ~std::size_t{15u}};
  CudaStagedBlock block{
      .host = workspace->upload.reserve(missingAt + sizeof(std::uint32_t)),
      .device = workspace->device.reserve(sizing.cursor)};
  stage(block);
  auto* nMissing = reinterpret_cast<std::uint32_t*>(block.host + missingAt);
  if (timing != nullptr) {
    timing->hostPreUs = microseconds(Clock::now() - hostPreStart);
  }

  // The copy, then the payload kernel, which counts towards the kernels
  std::array<cudaEvent_t, 3> uploadMarks{};
  if (timing != nullptr) {
    for (cudaEvent_t& mark : uploadMarks) {
      ACTS_CUDA_CHECK(cudaEventCreate(&mark));
    }
    ACTS_CUDA_CHECK(cudaEventRecord(uploadMarks[0], stream));
  }
  ACTS_CUDA_CHECK(cudaMemcpyAsync(block.device, block.host, uploadBytes,
                                  cudaMemcpyHostToDevice, stream));
  if (timing != nullptr) {
    ACTS_CUDA_CHECK(cudaEventRecord(uploadMarks[1], stream));
  }
  buildCudaHitPayloads(deviceSpacePoints, used,
                       CudaSurfaceTableArrays{.geometryId = surfaces.geometryId,
                                              .rotation = surfaces.rotation,
                                              .nSurfaces = surfaces.nSurfaces},
                       hits.deviceArrays(), deviceMissing, stream);
  ACTS_CUDA_CHECK(cudaMemcpyAsync(nMissing, deviceMissing,
                                  sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                                  stream));
  if (timing != nullptr) {
    ACTS_CUDA_CHECK(cudaEventRecord(uploadMarks[2], stream));
  }

  CudaPatternFinderSizes sizes{m_cfg.sizes};
  if (m_arenaBytes > 0u) {
    sizes.workspaceBytes = m_arenaBytes;
  }
  // Waits for the stream, so the count of hits without a surface is there
  const CudaPatternRecordContainer event{findCudaPatternsInEta(
      hits, lists, makeCudaPatternFinderConfig(m_cfg.finder, beamspot), sizes,
      stream, timing, &workspace->readback)};
  if (timing != nullptr) {
    float ms{0.f};
    ACTS_CUDA_CHECK(cudaEventElapsedTime(&ms, uploadMarks[0], uploadMarks[1]));
    timing->uploadUs = 1000. * ms;
    ACTS_CUDA_CHECK(cudaEventElapsedTime(&ms, uploadMarks[1], uploadMarks[2]));
    timing->kernelsUs += 1000. * ms;
    for (cudaEvent_t& mark : uploadMarks) {
      ACTS_CUDA_CHECK(cudaEventDestroy(mark));
    }
  }
  if (*nMissing != 0u) {
    throw std::runtime_error(std::format(
        "CudaGlobalPatternFinder: {} hits have no measurement surface",
        *nMissing));
  }
  if (const std::uint8_t overflow{event.overflow(0u)}; overflow != 0u) {
    ACTS_WARNING(__func__ << "() Patterns did not fit the device buffers "
                          << "(CudaPatternOverflow bits "
                          << static_cast<int>(overflow)
                          << "); the seeds that did not fit have no patterns");
  }

  // The payloads of the hits of the kept patterns, on the host, for the
  // phi-only hits and the conversion
  const auto hostPostStart = Clock::now();
  std::vector<std::optional<HitPayload>> payloads(usedPointer.size());
  auto payloadOf = [&](std::uint32_t row) -> const HitPayload& {
    std::optional<HitPayload>& payload{payloads[row]};
    if (!payload) {
      const std::uint32_t k{usedBucket[row]};
      payload.emplace(gctx, usedPointer[row], &spacePoints[k],
                      bucketTransform[k],
                      &measurementSurface(geometry, *usedPointer[row]));
    }
    return *payload;
  };
  MuonGlobalPatternContainer output{toMuonGlobalPatterns(
      finishPatterns(gctx, payloadOf, event, onlyPhiProvider, beamspot))};
  ACTS_DEBUG(__func__ << "() Found " << output.size() << " patterns from "
                      << lists.nSeeds() << " seeds");
  if (timing != nullptr) {
    timing->hostPostUs = microseconds(Clock::now() - hostPostStart);
  }
  return output;
}

CudaGlobalPatternFinder::CudaGlobalPatternFinder(
    Config cfg, std::unique_ptr<const Acts::Logger> logger)
    : m_cfg{std::move(cfg)}, m_logger{std::move(logger)} {}

std::vector<CudaGlobalPatternFinder::OutputPattern>
CudaGlobalPatternFinder::finishPatterns(
    const Acts::GeometryContext& gctx, const PayloadOf& payloadOf,
    const CudaPatternRecordContainer& event,
    const OnlyPhiHitsProvider& onlyPhiProvider,
    const BeamspotInfo& beamspot) const {
  std::vector<PatternStateAux> patterns{};
  for (std::uint32_t k = 0u; k < event.nPatterns(0u); ++k) {
    const std::size_t slot{event.slot(0u, k)};
    if (event.pattern(slot).status == CudaPatternStatus::ok) {
      patterns.push_back(
          makePatternState(gctx, payloadOf, event, slot, beamspot));
    }
  }

  if (m_cfg.finder.searchOnlyPhiHits) {
    addPhiOnlyHits(gctx, beamspot, onlyPhiProvider, patterns);
  }

  // GlobalPatternFinder::convertToPattern
  std::vector<OutputPattern> output{};
  output.reserve(patterns.size());
  for (PatternStateAux& pat : patterns) {
    OutputPattern& out{output.emplace_back(pat.sector.sector())};
    for (std::size_t g = 0u; g < PatternTopology::nGroups; ++g) {
      for (const auto& hit : pat.hitsPerGroup[g]) {
        out.hitsPerGroup[g].push_back(hit.hitPtr);
      }
    }
    out.phiOnlyHits = std::move(pat.phiOnlyHits);
    out.meanNormResidual2 = pat.getMeanResidual2();
    out.patTheta = pat.patTheta;
    out.patPhi = pat.patPhi;
    out.nPrecisionLayers = static_cast<std::uint8_t>(pat.nPrecisionLayers);
    out.nTriggerLayers = static_cast<std::uint8_t>(pat.nTriggerLayers);
    out.nPhiLayers = static_cast<std::uint8_t>(pat.nPhiLayers);
  }
  return output;
}

CudaGlobalPatternFinder::PatternStateAux
CudaGlobalPatternFinder::makePatternState(
    const Acts::GeometryContext& gctx, const PayloadOf& payloadOf,
    const CudaPatternRecordContainer& event, std::size_t slot,
    const BeamspotInfo& beamspot) const {
  using OrderedHit = PatternStateAux::OrderedHit;
  using BendPlaneCov = PatternStateAux::BendPlaneCov;
  const CudaPatternRecordRow row{event.pattern(slot)};
  auto orderedHit = [&](std::size_t i) {
    return OrderedHit{&payloadOf(event.hitIndex(slot, i)),
                      event.globLayer(slot, i)};
  };
  PatternStateAux pat{gctx,     orderedHit(0u), row.sector,
                      beamspot, &m_cfg.finder,  m_logger.get()};
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
  pat.sumNormResidual2 = row.sumNormResidual2;
  pat.patPhi = row.patPhi;
  pat.patTheta = row.patTheta;
  pat.patPhiOffset = row.patPhiOffset;
  pat.bendPlaneCov[Acts::toUnderlying(BendPlaneCov::ePhiPhi)] =
      row.bendPlaneCovPhiPhi;
  pat.bendPlaneCov[Acts::toUnderlying(BendPlaneCov::eSS)] = row.bendPlaneCovSS;
  pat.bendPlaneCov[Acts::toUnderlying(BendPlaneCov::ePhiS)] =
      row.bendPlaneCovPhiS;
  pat.nDoF = row.nDoF;
  pat.lastHitTestRes.residual = row.lastResidual;
  pat.lastHitTestRes.sigma = row.lastResSigma;
  pat.lastHitTestRes.nDof = row.lastNDoF;
  pat.nPrecisionLayers = row.nPrecisionLayers;
  pat.nTriggerLayers = row.nTriggerLayers;
  pat.nPhiLayers = row.nPhiLayers;
  pat.useBeamspot = (row.flags & CudaPatternFlags::useBeamspot) != 0u;
  pat.needLineUpdate = (row.flags & CudaPatternFlags::needLineUpdate) != 0u;
  return pat;
}

void CudaGlobalPatternFinder::addPhiOnlyHits(
    const Acts::GeometryContext& gctx, const BeamspotInfo& beamspot,
    const OnlyPhiHitsProvider& onlyPhiProvider,
    std::vector<PatternStateAux>& patterns) const {
  using OrderedHit = PatternStateAux::OrderedHit;
  // GlobalPatternFinder::isPhiCompatible
  auto isPhiCompatible = [&](const PatternStateAux& pat,
                             const HitPayload& hit) {
    const Acts::Vector3 pos{hit.globalPosition(gctx)};
    if (pat.nPhiLayers > 0) {
      return !(std::abs(pat.angleToBendPlane(pos)) > m_cfg.finder.phiTolerance);
    }
    return pat.sector.insideSector(Acts::VectorHelpers::phi(pos));
  };
  // The pattern line in the group of a phi hit, from its eta hits or from
  // the closest eta hits around it
  auto computePatternLineInGroup = [&](PatternStateAux& pat,
                                       const HitPayload& phiHit) -> bool {
    const auto phiHitGroup{PatternTopology::groupIndex(phiHit)};
    const auto& groupHits{pat.hitsPerGroup[phiHitGroup]};

    // useBeamspot flags whether the line could be drawn from the hits
    pat.useBeamspot = true;
    pat.needLineUpdate = true;

    if (groupHits.size() > 1) {
      const auto [minIt, maxIt]{std::ranges::minmax_element(
          groupHits, {}, [](const OrderedHit& c) { return c.globLayer; })};
      pat.lineAnchorHit = *minIt;
      pat.lastInsertedHit = *maxIt;
      pat.updateLineParameters(gctx, BeamspotInfo{});
    }
    if (pat.useBeamspot && !groupHits.empty()) {
      pat.needLineUpdate = true;
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
      pat.needLineUpdate = true;
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

  std::vector<PatternStateAux> survivingPatterns{};
  survivingPatterns.reserve(patterns.size());
  for (PatternStateAux& pat : patterns) {
    std::optional<PatternTopology::GroupIdx> patternLineGroup{std::nullopt};

    auto projOntoPhiPlane = [&pat](const Acts::Vector3& pos) -> Acts::Vector3 {
      return pos - (pos.dot(pat.bendPlaneNorm) - pat.patPhiOffset) *
                       pat.bendPlaneNorm;
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
        if (!isPhiCompatible(pat, newHit)) {
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
        // One phi hit per layer, the one closest to the bending plane
        if (const auto it = std::ranges::find_if(
                pat.phiOnlyHits,
                [&](const HitPayload& h) {
                  return PatternTopology::sameLayer(h, newHit);
                });
            it != pat.phiOnlyHits.end()) {
          if (std::abs(pat.angleToBendPlane(it->globalPosition(gctx))) <
              std::abs(pat.angleToBendPlane(newHitPos))) {
            continue;
          }
          *it = std::move(newHit);
        } else {
          pat.phiOnlyHits.push_back(std::move(newHit));
          pat.nPhiLayers++;
        }
        pat.updatePatternPhi(gctx, beamspot);
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
  patCfg.maxThetaSizeOverlap = 2. * m_cfg.thetaSearchWindow;
  patCfg.phiTolerance = m_cfg.phiTolerance;
  patCfg.loosenAcceptanceFactor = 2.5;
  patCfg.minStripEtaLayers = m_cfg.minTriggerLayers;
  patCfg.minPrecisionLayers = m_cfg.minPrecisionLayers;
  patCfg.minPhiLayers = m_cfg.minPhiLayers;
  patCfg.minGroupLayers = m_cfg.minStationLayers;
  patCfg.meanNormRes2Cut = m_cfg.meanNormRes2Cut;
  patCfg.maxSeedAttempts = m_cfg.maxSeedAttempts;
  patCfg.maxMissLayersInGroup = m_cfg.maxMissLayersInStation;
  patCfg.minHitDistance4Line = m_cfg.minHitDistance4Line;
  finderCfg.sizes =
      CudaPatternFinderSizes{.workspaceBytes = m_cfg.workspaceBytes};
  finderCfg.trackingGeometry = m_cfg.trackingGeometry;
  finderCfg.useMdtHits = m_cfg.useMdtHits;
  finderCfg.workspaceFraction = m_cfg.workspaceFraction;
  m_finder = std::make_unique<CudaGlobalPatternFinder>(
      std::move(finderCfg), this->logger().clone("CudaGlobalPatternFinder"));
}

ProcessCode CudaGlobalPatternFinderAlgorithm::execute(
    const AlgorithmContext& ctx) const {
  const MuonSpacePointContainer& inSpacePoints{m_inSpacePoints(ctx)};
  const Acts::GeometryContext& gctx{ctx.recoGeoContext};

  CudaPatternFinderTiming timing{};
  GlobalPatternFinder_t::BeamspotInfo beamSpot{};
  beamSpot.position = Acts::Vector3::Zero();
  beamSpot.radius = m_cfg.beamSpotRadius;
  beamSpot.length = m_cfg.beamSpotLength;

  auto found{m_finder->findPatterns(gctx, inSpacePoints, *m_seedSelector,
                                    m_onlyPhiProvider, beamSpot, &timing)};
  ACTS_DEBUG("Found " << found.size() << " patterns from " << timing.nSeeds
                      << " seeds");
  m_outPatterns(ctx, std::move(found));
  if (m_outTiming.isInitialized()) {
    m_outTiming(ctx, std::move(timing));
  }
  return ProcessCode::SUCCESS;
}

MuonGlobalPatternContainer toMuonGlobalPatterns(
    const std::vector<GlobalPatternFinder_t::OutputPattern>& found) {
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
    pattern.sector = candidate.sector.sector();
    pattern.nPrecisionLayers = candidate.nPrecisionLayers;
    pattern.nTriggerLayers = candidate.nTriggerLayers;
    pattern.nPhiLayers = candidate.nPhiLayers;
    pattern.meanNormResidual2 = candidate.meanNormResidual2;
  }
  return patterns;
}

}  // namespace ActsExamples
