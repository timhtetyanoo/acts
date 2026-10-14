// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

/// The host side of version 2 of the GPU pattern finding: the search tree over
/// the global positions, the candidate lists, the upload of the space points
/// and the payloads of the hits of the kept patterns.

#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/KDTree.hpp"
#include "Acts/Utilities/VectorHelpers.hpp"
#include "ActsExamples/TrackFinding/CudaPatternFinder.hpp"
#include "ActsExamples/Utilities/CudaStream.hpp"
#include "ActsExamples/Utilities/CudaUtilities.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <format>
#include <optional>
#include <stdexcept>

namespace ActsExamples {

namespace {

/// A device buffer from the memory pool, released with the object once the
/// work of the event has completed
template <typename T>
class PoolBuffer {
 public:
  PoolBuffer(std::size_t size, cudaStream_t stream) {
    if (size > 0u) {
      ACTS_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&m_data),
                                      size * sizeof(T), stream));
    }
  }
  PoolBuffer(const std::vector<T>& host, cudaStream_t stream)
      : PoolBuffer(host.size(), stream) {
    if (!host.empty()) {
      ACTS_CUDA_CHECK(cudaMemcpyAsync(m_data, host.data(),
                                      host.size() * sizeof(T),
                                      cudaMemcpyHostToDevice, stream));
    }
  }
  PoolBuffer(const PoolBuffer&) = delete;
  PoolBuffer& operator=(const PoolBuffer&) = delete;
  ~PoolBuffer() {
    if (m_data != nullptr) {
      (void)cudaFreeAsync(m_data, cudaStreamPerThread);
    }
  }
  T* get() const { return m_data; }

 private:
  T* m_data{nullptr};
};

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
          "CudaGlobalPatternFinder: version 2 needs the tracking geometry");
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

MuonGlobalPatternContainer CudaGlobalPatternFinder::findPatternsV2(
    const Acts::GeometryContext& gctx,
    const MuonSpacePointContainer& spacePoints,
    const SeedSelector& seedSelector,
    const OnlyPhiHitsProvider& onlyPhiProvider, const BeamspotInfo& beamspot,
    CudaPatternFinderTiming* timing) const {
  using Clock = std::chrono::steady_clock;
  auto microseconds = [](Clock::duration d) {
    return std::chrono::duration<double, std::micro>(d).count();
  };
  using enum ExpandedSector::SectorProjector;
  const SurfaceTable& surfaces{surfaceTable(gctx)};
  const Acts::TrackingGeometry& geometry{*m_cfg.trackingGeometry};

  // The global positions of the hits the finding uses and the search tree
  // over them, as GlobalPatternFinderAlgorithm::constructTree
  const auto treeStart = Clock::now();
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
    std::optional<std::uint16_t> msSector{};
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
      if (!msSector) {
        msSector = sp.id().sector();
      }
      const ExpandedSector hitExpSector{Acts::VectorHelpers::phi(pos)};
      // A hit close to a sector border also enters the neighbouring sector
      for (const ExpandedSector::SectorProjector proj :
           {leftOverlap, center, rightOverlap}) {
        const ExpandedSector expSect{*msSector, proj};
        if (proj != center && sp.id().measuresPhi() &&
            expSect != hitExpSector) {
          continue;
        }
        // Projected onto the plane along the sector radial direction
        const Acts::Vector3 planeNormal{expSect.normalDir()};
        const double projR{
            sp.id().measuresPhi()
                ? Acts::VectorHelpers::perp(pos)
                : Acts::VectorHelpers::perp(
                      Acts::Vector3{pos - pos.dot(planeNormal) * planeNormal})};
        std::array<double, 2> coords{};
        using HitCoords = GlobalPatternFinder_t::HitCoords;
        coords[Acts::toUnderlying(HitCoords::eTheta)] =
            std::atan2(projR, pos.z());
        coords[Acts::toUnderlying(HitCoords::eSector)] = expSect.sector();
        treeEntries.emplace_back(coords, row);
      }
    }
    bucketEnd[k] = static_cast<std::uint32_t>(usedPointer.size());
  }
  const RowTree tree{std::move(treeEntries)};
  if (timing != nullptr) {
    timing->treeUs = microseconds(Clock::now() - treeStart);
  }

  // The candidate lists, as makeCudaCandidateLists, and the flat space points
  const auto fillStart = Clock::now();
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
  // The used hits only, row r being space point r; every bucket keeps its
  // number, those without used hits as empty ranges
  CudaMuonSpacePointContainer flatSpacePoints{usedPointer.size()};
  for (std::size_t row = 0u; row < usedPointer.size(); ++row) {
    const MuonSpacePoint& sp{*usedPointer[row]};
    flatSpacePoints.setGeometryId(row, sp.geometryId().value());
    flatSpacePoints.setId(row, sp.id().toInt());
    flatSpacePoints.defineCoordinates(row, sp.localPosition(),
                                      sp.sensorDirection(), sp.toNextSensor());
    flatSpacePoints.setRadius(row, sp.driftRadius());
    flatSpacePoints.setTime(row, sp.time());
    const auto& covariance = sp.covariance();
    flatSpacePoints.setCovariance(row, covariance[0], covariance[1],
                                  covariance[2]);
  }
  for (std::size_t k = 0u; k < spacePoints.size(); ++k) {
    flatSpacePoints.addBucket(bucketBegin[k], bucketEnd[k]);
  }
  if (timing != nullptr) {
    timing->fillUs = microseconds(Clock::now() - fillStart);
  }

  // Up to the device: the used hits, their positions, the bucket rotations
  // and the lists; the payloads are built there
  CudaStream stream;
  std::array<cudaEvent_t, 2> uploadMarks{};
  if (timing != nullptr) {
    for (cudaEvent_t& mark : uploadMarks) {
      ACTS_CUDA_CHECK(cudaEventCreate(&mark));
    }
    ACTS_CUDA_CHECK(cudaEventRecord(uploadMarks[0], stream.get()));
  }
  flatSpacePoints.moveToDevice(stream.get());
  const PoolBuffer<double> devicePositionX{positionX, stream.get()};
  const PoolBuffer<double> devicePositionY{positionY, stream.get()};
  const PoolBuffer<double> devicePositionZ{positionZ, stream.get()};
  const PoolBuffer<double> deviceBucketRotation{bucketRotation, stream.get()};
  lists.moveToDevice(stream.get());
  CudaHitPayloadContainer hits{};
  hits.allocateDevice(usedPointer.size(), stream.get());
  const PoolBuffer<std::uint32_t> missingSurfaces{1u, stream.get()};
  ACTS_CUDA_CHECK(cudaMemsetAsync(missingSurfaces.get(), 0,
                                  sizeof(std::uint32_t), stream.get()));
  buildCudaHitPayloads(
      flatSpacePoints.deviceArrays(),
      CudaUsedHitArrays{
          .positionX = devicePositionX.get(),
          .positionY = devicePositionY.get(),
          .positionZ = devicePositionZ.get(),
          .bucketRotation = deviceBucketRotation.get(),
          .nHits = static_cast<std::uint32_t>(usedPointer.size()),
          .nBuckets = static_cast<std::uint32_t>(spacePoints.size())},
      CudaSurfaceTableArrays{.geometryId = surfaces.geometryId,
                             .rotation = surfaces.rotation,
                             .nSurfaces = surfaces.nSurfaces},
      hits.deviceArrays(), missingSurfaces.get(), stream.get());
  if (timing != nullptr) {
    ACTS_CUDA_CHECK(cudaEventRecord(uploadMarks[1], stream.get()));
  }

  CudaPatternFinderSizes sizes{m_cfg.sizes};
  if (m_arenaBytes > 0u) {
    sizes.workspaceBytes = m_arenaBytes;
  }
  const CudaPatternRecordContainer event{findCudaPatternsInEtaV2(
      hits, lists, makeCudaPatternFinderConfig(m_cfg.finder, beamspot), sizes,
      stream.get(), timing)};
  std::uint32_t nMissing{0u};
  ACTS_CUDA_CHECK(cudaMemcpyAsync(&nMissing, missingSurfaces.get(),
                                  sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                                  stream.get()));
  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  if (timing != nullptr) {
    float ms{0.f};
    ACTS_CUDA_CHECK(cudaEventElapsedTime(&ms, uploadMarks[0], uploadMarks[1]));
    timing->uploadUs = 1000. * ms;
    for (cudaEvent_t& mark : uploadMarks) {
      ACTS_CUDA_CHECK(cudaEventDestroy(mark));
    }
  }
  if (nMissing != 0u) {
    throw std::runtime_error(std::format(
        "CudaGlobalPatternFinder: {} hits have no measurement surface",
        nMissing));
  }
  if (const std::uint8_t overflow{event.overflow(0u)}; overflow != 0u) {
    ACTS_WARNING(__func__ << "() Patterns did not fit the device buffers "
                          << "(CudaPatternOverflow bits "
                          << static_cast<int>(overflow)
                          << "); the seeds that did not fit have no patterns");
  }

  // The payloads of the hits of the kept patterns, on the host, for the
  // phi-only hits and the conversion
  const auto tailStart = Clock::now();
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
    timing->tailUs = microseconds(Clock::now() - tailStart);
  }
  return output;
}

}  // namespace ActsExamples
