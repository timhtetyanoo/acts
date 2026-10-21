// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "ActsExamples/EventData/MuonSpacePoint.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace ActsExamples {

/// @brief Station name, sector and side of a measurement as one key
inline std::uint32_t stationKey(const MuonSpacePoint::MuonId& id) noexcept {
  const auto station = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(Acts::toUnderlying(id.msStation())));
  const auto side = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(Acts::toUnderlying(id.side())));

  return (side << 24u) | (station << 16u) | id.sector();
}

/// @brief Page-locked host memory reused between events; grows, never
///        shrinks
class CudaPinnedBuffer {
 public:
  CudaPinnedBuffer() = default;
  CudaPinnedBuffer(const CudaPinnedBuffer&) = delete;
  CudaPinnedBuffer& operator=(const CudaPinnedBuffer&) = delete;
  ~CudaPinnedBuffer();
  /// @brief At least `bytes` bytes; the content is not kept when it grows
  char* reserve(std::size_t bytes);
  char* data() const noexcept { return m_data; }

 private:
  char* m_data = nullptr;
  std::size_t m_size = 0;
};

/// @brief Device memory reused between events; grows, never shrinks
class CudaDeviceBuffer {
 public:
  CudaDeviceBuffer() = default;
  CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
  CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;
  ~CudaDeviceBuffer();
  /// @brief At least `bytes` bytes; the content is not kept when it grows
  char* reserve(std::size_t bytes);
  char* data() const noexcept { return m_data; }

 private:
  char* m_data = nullptr;
  std::size_t m_size = 0;
};

/// @brief Columns at the same offsets in a host and a device block, so one
///        copy moves them all. With null blocks it only counts the bytes.
struct CudaStagedBlock {
  char* host = nullptr;
  char* device = nullptr;
  std::size_t cursor = 0;

  /// @brief Room for n elements of T, aligned; returns the host and the
  ///        device address, null while counting
  template <typename T>
  std::pair<T*, T*> take(std::size_t n) {
    const std::size_t at{cursor};
    cursor += (n * sizeof(T) + 15u) & ~std::size_t{15u};
    if (host == nullptr || n == 0u) {
      return {nullptr, nullptr};
    }
    return {reinterpret_cast<T*>(host + at), reinterpret_cast<T*>(device + at)};
  }
};

/// @brief Bits of the flags column of the hit payloads
struct CudaHitFlags {
  /// @brief Drift tube measurement, the sensor is the wire
  static constexpr std::uint8_t straw = 1u << 0u;
  /// @brief The hit measures the precision coordinate
  static constexpr std::uint8_t measuresEta = 1u << 1u;
  /// @brief The hit measures the non-precision coordinate as well
  static constexpr std::uint8_t measuresPhi = 1u << 2u;
  static constexpr std::uint8_t precision = 1u << 3u;
  /// @brief Tilted strips: the phi measurement direction differs from the
  ///        sensor direction
  static constexpr std::uint8_t tiltedStrips = 1u << 4u;
};

/// @brief The quantities the pattern finder reads from a hit
struct CudaHitPayloadRow {
  Acts::Vector3 position{Acts::Vector3::Zero()};
  /// @brief Global direction of the wire or of the strip
  Acts::Vector3 sensorDirection{Acts::Vector3::Zero()};
  /// Zero for a straw
  Acts::Vector3 etaMeasurementDirection{Acts::Vector3::Zero()};
  /// Zero unless the strip also measures phi
  Acts::Vector3 phiMeasurementDirection{Acts::Vector3::Zero()};

  /// Breaks ties within a layer and tells measurements apart
  double localPositionY{0.};
  double etaCovariance{0.};
  double phiCovariance{0.};
  /// Drift radius squared plus the bending covariance; zero for a strip
  double driftCovariance{0.};

  std::uint64_t geometryId{0u};
  std::uint32_t stationKey{0u};
  std::uint8_t station{0u};
  std::uint8_t locLayer{0u};
  std::uint8_t flags{0u};
};

/// @brief Non-owning device view of the hit payloads, one column per field
///        of CudaHitPayloadRow
struct CudaHitPayloadArrays {
  double* positionX = nullptr;
  double* positionY = nullptr;
  double* positionZ = nullptr;

  double* sensorDirectionX = nullptr;
  double* sensorDirectionY = nullptr;
  double* sensorDirectionZ = nullptr;

  double* etaMeasurementDirectionX = nullptr;
  double* etaMeasurementDirectionY = nullptr;
  double* etaMeasurementDirectionZ = nullptr;

  double* phiMeasurementDirectionX = nullptr;
  double* phiMeasurementDirectionY = nullptr;
  double* phiMeasurementDirectionZ = nullptr;

  double* localPositionY = nullptr;
  double* etaCovariance = nullptr;
  double* phiCovariance = nullptr;
  double* driftCovariance = nullptr;

  std::uint64_t* geometryId = nullptr;

  std::uint32_t* stationKey = nullptr;

  std::uint8_t* station = nullptr;
  std::uint8_t* locLayer = nullptr;
  std::uint8_t* flags = nullptr;
};

struct CudaHitPayloadHostData {
  std::vector<double> positionX;
  std::vector<double> positionY;
  std::vector<double> positionZ;

  std::vector<double> sensorDirectionX;
  std::vector<double> sensorDirectionY;
  std::vector<double> sensorDirectionZ;

  std::vector<double> etaMeasurementDirectionX;
  std::vector<double> etaMeasurementDirectionY;
  std::vector<double> etaMeasurementDirectionZ;

  std::vector<double> phiMeasurementDirectionX;
  std::vector<double> phiMeasurementDirectionY;
  std::vector<double> phiMeasurementDirectionZ;

  std::vector<double> localPositionY;
  std::vector<double> etaCovariance;
  std::vector<double> phiCovariance;
  std::vector<double> driftCovariance;

  std::vector<std::uint64_t> geometryId;

  std::vector<std::uint32_t> stationKey;

  std::vector<std::uint8_t> station;
  std::vector<std::uint8_t> locLayer;
  std::vector<std::uint8_t> flags;
};

/// @brief Hit payloads on the host and the device. Row i is the i-th payload
///        of the search tree's storage, so tree entries map to rows by
///        pointer arithmetic.
class CudaHitPayloadContainer {
 public:
  using size_type = std::size_t;

  CudaHitPayloadContainer() = default;

  explicit CudaHitPayloadContainer(size_type size);

  CudaHitPayloadContainer(const CudaHitPayloadContainer&) = delete;
  CudaHitPayloadContainer& operator=(const CudaHitPayloadContainer&) = delete;

  CudaHitPayloadContainer(CudaHitPayloadContainer&& other) noexcept;
  CudaHitPayloadContainer& operator=(CudaHitPayloadContainer&& other) noexcept;

  ~CudaHitPayloadContainer() noexcept;

  size_type size() const noexcept { return m_size; }

  bool empty() const noexcept { return size() == 0; }

  void setHit(size_type index, const CudaHitPayloadRow& row);

  void moveToDevice(cudaStream_t stream);

  /// @brief Places device-only columns, filled by a kernel, in a
  ///        caller-owned block. The host accessors are not usable afterwards.
  void placeOnDevice(size_type size, CudaStagedBlock& block);

  void clearDevice() noexcept;

  bool isOnDevice() const noexcept { return m_onDevice; }

  CudaHitPayloadArrays deviceArrays() const noexcept { return m_device; }

 private:
  size_type m_size = 0;
  CudaHitPayloadHostData m_host{};
  CudaHitPayloadArrays m_device{};
  void* m_block = nullptr;
  bool m_onDevice = false;

  void checkIndex(size_type index) const;
};

/// @brief Non-owning device view of the candidate lists. Seed s owns
///        candidateIndices[candidateOffsets[s], candidateOffsets[s + 1]),
///        which the seed checks sort in place.
struct CudaCandidateArrays {
  std::uint32_t* seedHitIndex = nullptr;
  double* seedSector = nullptr;
  double* seedTheta = nullptr;

  std::uint32_t* candidateOffsets = nullptr;
  std::uint32_t* candidateIndices = nullptr;

  // Outputs of the seed checks
  std::uint8_t* seedStatus = nullptr;
  std::uint16_t* seedPosition = nullptr;
  std::uint16_t* candidateGlobLayer = nullptr;
  /// Group in bits 0-2, measures phi in bit 3, precision in bit 4
  std::uint8_t* candidateInfo = nullptr;

  std::uint32_t nSeeds = 0;
  std::uint32_t nCandidates = 0;

  __host__ __device__ std::uint32_t begin(std::uint32_t seed) const noexcept {
    return candidateOffsets[seed];
  }
  __host__ __device__ std::uint32_t end(std::uint32_t seed) const noexcept {
    return candidateOffsets[seed + 1u];
  }
};

struct CudaCandidateHostData {
  std::vector<std::uint32_t> seedHitIndex;
  std::vector<double> seedSector;
  std::vector<double> seedTheta;

  /// nSeeds() + 1 entries, starting as {0}
  std::vector<std::uint32_t> candidateOffsets{0};
  std::vector<std::uint32_t> candidateIndices;

  std::vector<std::uint8_t> seedStatus;
  std::vector<std::uint16_t> seedPosition;
  std::vector<std::uint16_t> candidateGlobLayer;
  std::vector<std::uint8_t> candidateInfo;
};

/// @brief Seed status, in the order the cuts are applied
struct CudaSeedStatus {
  static constexpr std::uint8_t pending = 0u;
  static constexpr std::uint8_t accepted = 1u;
  /// @brief Fewer candidates than minStripEtaLayers + minPrecisionLayers
  static constexpr std::uint8_t tooFewHits = 2u;
  /// @brief Fewer than minGroups groups with minGroupLayers candidates
  static constexpr std::uint8_t tooFewGroups = 3u;
  /// @brief Fewer layers than minStripEtaLayers + minPrecisionLayers
  static constexpr std::uint8_t tooFewLayers = 4u;
  /// @brief The list does not fit the 16-bit positions; the seed is not built
  static constexpr std::uint8_t overflow = 5u;
};

/// @brief Per-seed candidate lists: the payload rows of each seed's range
///        search, flat, with the start of each seed
class CudaCandidateListContainer {
 public:
  using size_type = std::size_t;

  CudaCandidateListContainer() = default;

  CudaCandidateListContainer(const CudaCandidateListContainer&) = delete;
  CudaCandidateListContainer& operator=(const CudaCandidateListContainer&) =
      delete;

  CudaCandidateListContainer(CudaCandidateListContainer&& other) noexcept;
  CudaCandidateListContainer& operator=(
      CudaCandidateListContainer&& other) noexcept;

  ~CudaCandidateListContainer() noexcept;

  size_type nSeeds() const noexcept {
    return m_deviceOnly ? m_device.nSeeds : m_host.seedHitIndex.size();
  }

  size_type nCandidates() const noexcept {
    return m_deviceOnly ? m_device.nCandidates : m_host.candidateIndices.size();
  }

  size_type nCandidates(size_type seed) const;

  bool empty() const noexcept { return nSeeds() == 0; }

  /// @brief Appends a seed and the payload rows of its candidates
  void addSeed(std::uint32_t seedHitIndex, double sector, double theta,
               std::span<const std::uint32_t> hitIndices);

  void addSeed(std::uint32_t seedHitIndex, double sector, double theta,
               std::initializer_list<std::uint32_t> hitIndices) {
    addSeed(
        seedHitIndex, sector, theta,
        std::span<const std::uint32_t>{hitIndices.begin(), hitIndices.size()});
  }

  size_type candidateBegin(size_type seed) const;

  size_type candidateEnd(size_type seed) const;

  std::span<const std::uint32_t> candidateHitIndices(size_type seed) const;

  // Results of the seed checks
  std::uint8_t seedStatus(size_type seed) const;

  std::uint16_t seedPosition(size_type seed) const;

  std::span<const std::uint16_t> candidateGlobLayers(size_type seed) const;

  /// @brief Uploads the inputs and zeroes the outputs; asynchronous
  void moveToDevice(cudaStream_t stream);

  /// @brief Device-only lists for a kernel to fill; only the counts are
  ///        usable on the host afterwards
  void allocateDevice(size_type nSeeds, size_type nCandidates,
                      cudaStream_t stream);

  void moveToHost(cudaStream_t stream);

  void clearDevice() noexcept;

  bool isOnDevice() const noexcept { return m_onDevice; }

  CudaCandidateArrays deviceArrays() const noexcept { return m_device; }

 private:
  CudaCandidateHostData m_host{};
  CudaCandidateArrays m_device{};
  void* m_block = nullptr;
  bool m_onDevice = false;
  bool m_deviceOnly = false;

  void checkSeed(size_type seed) const;
};

/// @brief Marks an unset hit reference
constexpr std::uint16_t cudaInvalidHitPosition = 0xFFFFu;

/// @brief Bits of the flags column of a pattern record
struct CudaPatternFlags {
  static constexpr std::uint8_t useBeamspot = 1u << 0u;
  static constexpr std::uint8_t needLineUpdate = 1u << 1u;
  /// Lost against an overlapping pattern
  static constexpr std::uint8_t isOverlap = 1u << 3u;
};

/// @brief Status of a pattern slot
struct CudaPatternStatus {
  static constexpr std::uint8_t empty = 0u;
  static constexpr std::uint8_t ok = 1u;
  static constexpr std::uint8_t rejected = 2u;
};

/// @brief What did not fit
struct CudaPatternOverflow {
  static constexpr std::uint8_t patterns = 1u << 0u;
  static constexpr std::uint8_t hits = 1u << 1u;
  static constexpr std::uint8_t eventPatterns = 1u << 2u;
};

/// @brief One pattern record
struct CudaPatternRecordRow {
  std::int8_t sector{0};
  double patPhi{0.};
  double patTheta{0.};
  double patPhiOffset{0.};
  double bendPlaneCovPhiPhi{0.};
  double bendPlaneCovSS{0.};
  double bendPlaneCovPhiS{0.};
  double sumNormResidual2{0.};
  std::uint32_t nDoF{0u};
  double lastResidual{0.};
  double lastResSigma{0.};
  std::uint8_t lastNDoF{0u};
  /// Radius at which the overlap test compares the phi of two patterns
  double anchorRadius{0.};
  Acts::Vector3 linePos{Acts::Vector3::Zero()};
  Acts::Vector3 lineDir{Acts::Vector3::Zero()};
  double leverArm{0.};
  Acts::Vector3 bendPlaneNorm{Acts::Vector3::Zero()};
  std::uint8_t nPrecisionLayers{0u};
  std::uint8_t nTriggerLayers{0u};
  std::uint8_t nPhiLayers{0u};
  std::uint32_t nHits{0u};
  // Indices into the pattern's own hit list
  std::uint16_t lastInsertedHit{cudaInvalidHitPosition};
  std::uint16_t prevLayerHit{cudaInvalidHitPosition};
  std::uint16_t lineAnchorHit{cudaInvalidHitPosition};
  std::uint8_t flags{0u};
  std::uint8_t status{CudaPatternStatus::empty};
};

/// @brief Non-owning device view of the pattern records. Entry e holds
///        capacityPerSeed pattern slots of maxHitsPerPattern hit slots each.
struct CudaPatternRecordArrays {
  std::int8_t* sector = nullptr;

  double* patPhi = nullptr;
  double* patTheta = nullptr;
  double* patPhiOffset = nullptr;
  double* bendPlaneCovPhiPhi = nullptr;
  double* bendPlaneCovSS = nullptr;
  double* bendPlaneCovPhiS = nullptr;
  std::uint32_t* nDoF = nullptr;
  std::uint8_t* lastNDoF = nullptr;
  double* anchorRadius = nullptr;

  double* sumNormResidual2 = nullptr;
  double* lastResidual = nullptr;
  double* lastResSigma = nullptr;

  double* linePosX = nullptr;
  double* linePosY = nullptr;
  double* linePosZ = nullptr;

  double* lineDirX = nullptr;
  double* lineDirY = nullptr;
  double* lineDirZ = nullptr;

  double* leverArm = nullptr;

  double* bendPlaneNormX = nullptr;
  double* bendPlaneNormY = nullptr;
  double* bendPlaneNormZ = nullptr;

  std::uint8_t* nPrecisionLayers = nullptr;
  std::uint8_t* nTriggerLayers = nullptr;
  std::uint8_t* nPhiLayers = nullptr;

  std::uint32_t* nHits = nullptr;

  std::uint16_t* lastInsertedHit = nullptr;
  std::uint16_t* prevLayerHit = nullptr;
  std::uint16_t* lineAnchorHit = nullptr;

  std::uint8_t* flags = nullptr;
  std::uint8_t* status = nullptr;

  std::uint32_t* hitIndex = nullptr;
  std::uint16_t* globLayer = nullptr;

  std::uint32_t* nPatterns = nullptr;
  std::uint8_t* overflow = nullptr;

  std::uint32_t nSeeds = 0;
  std::uint32_t capacityPerSeed = 0;
  std::uint32_t maxHitsPerPattern = 0;

  __host__ __device__ std::uint32_t slot(std::uint32_t seed,
                                         std::uint32_t k) const noexcept {
    return seed * capacityPerSeed + k;
  }
  __host__ __device__ std::uint32_t hitSlot(
      std::uint32_t patternSlot, std::uint32_t local) const noexcept {
    return patternSlot * maxHitsPerPattern + local;
  }
};

struct CudaPatternRecordHostData {
  std::vector<std::int8_t> sector;

  std::vector<double> patPhi;
  std::vector<double> patTheta;
  std::vector<double> patPhiOffset;
  std::vector<double> bendPlaneCovPhiPhi;
  std::vector<double> bendPlaneCovSS;
  std::vector<double> bendPlaneCovPhiS;
  std::vector<std::uint32_t> nDoF;
  std::vector<std::uint8_t> lastNDoF;
  std::vector<double> anchorRadius;

  std::vector<double> sumNormResidual2;
  std::vector<double> lastResidual;
  std::vector<double> lastResSigma;

  std::vector<double> linePosX;
  std::vector<double> linePosY;
  std::vector<double> linePosZ;

  std::vector<double> lineDirX;
  std::vector<double> lineDirY;
  std::vector<double> lineDirZ;

  std::vector<double> leverArm;

  std::vector<double> bendPlaneNormX;
  std::vector<double> bendPlaneNormY;
  std::vector<double> bendPlaneNormZ;

  std::vector<std::uint8_t> nPrecisionLayers;
  std::vector<std::uint8_t> nTriggerLayers;
  std::vector<std::uint8_t> nPhiLayers;

  std::vector<std::uint32_t> nHits;

  std::vector<std::uint16_t> lastInsertedHit;
  std::vector<std::uint16_t> prevLayerHit;
  std::vector<std::uint16_t> lineAnchorHit;

  std::vector<std::uint8_t> flags;
  std::vector<std::uint8_t> status;

  std::vector<std::uint32_t> hitIndex;
  std::vector<std::uint16_t> globLayer;

  std::vector<std::uint32_t> nPatterns;
  std::vector<std::uint8_t> overflow;
};

/// @brief Pattern records, written by the kernels: moveToDevice() allocates
///        and zeroes without uploading
class CudaPatternRecordContainer {
 public:
  using size_type = std::size_t;

  CudaPatternRecordContainer(size_type nSeeds, size_type capacityPerSeed,
                             size_type maxHitsPerPattern);

  /// @brief Fills the host columns from a host copy of the device block the
  ///        source columns live in: column c is read at
  ///        hostCopy + (c - deviceBase)
  void copyFromHostCopy(const CudaPatternRecordArrays& source,
                        const char* deviceBase, const char* hostCopy);

  CudaPatternRecordContainer(const CudaPatternRecordContainer&) = delete;
  CudaPatternRecordContainer& operator=(const CudaPatternRecordContainer&) =
      delete;

  CudaPatternRecordContainer(CudaPatternRecordContainer&& other) noexcept;
  CudaPatternRecordContainer& operator=(
      CudaPatternRecordContainer&& other) noexcept;

  ~CudaPatternRecordContainer() noexcept;

  size_type nSeeds() const noexcept { return m_nSeeds; }
  size_type capacityPerSeed() const noexcept { return m_capacityPerSeed; }
  size_type maxHitsPerPattern() const noexcept { return m_maxHitsPerPattern; }
  size_type nSlots() const noexcept { return m_nSeeds * m_capacityPerSeed; }
  bool empty() const noexcept { return nSlots() == 0; }

  size_type slot(size_type seed, size_type k) const;

  std::uint32_t nPatterns(size_type seed) const;
  void setNPatterns(size_type seed, std::uint32_t nPatterns);

  std::uint8_t overflow(size_type seed) const;
  void setOverflow(size_type seed, std::uint8_t overflow);

  CudaPatternRecordRow pattern(size_type patternSlot) const;

  std::uint32_t nHits(size_type patternSlot) const;

  std::uint32_t hitIndex(size_type patternSlot, size_type local) const;

  std::uint16_t globLayer(size_type patternSlot, size_type local) const;

  std::span<const std::uint32_t> hitIndices(size_type patternSlot) const;

  void clearDevice() noexcept;

  bool isOnDevice() const noexcept { return m_onDevice; }

  CudaPatternRecordArrays deviceArrays() const noexcept { return m_device; }

 private:
  CudaPatternRecordContainer() = default;

  size_type m_nSeeds = 0;
  size_type m_capacityPerSeed = 0;
  size_type m_maxHitsPerPattern = 0;
  CudaPatternRecordHostData m_host{};
  CudaPatternRecordArrays m_device{};
  void* m_block = nullptr;
  bool m_onDevice = false;

  size_type hitSlot(size_type patternSlot, size_type local) const;
  void checkSeed(size_type seed) const;
  void checkSlot(size_type patternSlot) const;
  void checkHit(size_type patternSlot, size_type local) const;
};

/// @brief Index of the bending coordinate in MuonSpacePoint::covariance()
constexpr std::size_t cudaHitEtaCovarianceIndex = 1u;
/// @brief Index of the non-bending coordinate
constexpr std::size_t cudaHitPhiCovarianceIndex = 0u;

/// @brief Templated so that this library does not depend on the track
///        finding library
/// @tparam Payload ActsExamples::HitPayload
template <typename Payload>
CudaHitPayloadRow makeCudaHitPayloadRow(const Acts::GeometryContext& gctx,
                                        const Payload& hit) {
  const MuonSpacePoint& sp{*hit.spacePoint()};

  CudaHitPayloadRow row{};
  row.position = hit.globalPosition(gctx);
  row.localPositionY = sp.localPosition().y();
  row.etaCovariance = sp.covariance()[cudaHitEtaCovarianceIndex];
  row.phiCovariance = sp.covariance()[cudaHitPhiCovarianceIndex];

  row.geometryId = sp.geometryId().value();
  row.stationKey = stationKey(sp.id());
  row.station = static_cast<std::uint8_t>(Acts::toUnderlying(hit.station));
  row.locLayer = hit.locLayer;

  if (sp.isStraw()) {
    row.flags |= CudaHitFlags::straw;
  }
  if (sp.id().measuresEta()) {
    row.flags |= CudaHitFlags::measuresEta;
  }
  if (sp.id().measuresPhi()) {
    row.flags |= CudaHitFlags::measuresPhi;
  }
  if (hit.isPrecision()) {
    row.flags |= CudaHitFlags::precision;
  }

  // Phi-only hits do not take part in the pattern building in eta, so they
  // need no sensor direction or variance
  if (!sp.id().measuresEta()) {
    return row;
  }

  row.sensorDirection = hit.globalSensorDirection(gctx);

  if (sp.isStraw()) {
    row.driftCovariance = Acts::square(sp.driftRadius()) +
                          sp.covariance()[cudaHitEtaCovarianceIndex];
    return row;
  }

  const auto [primary, secondary] = hit.stripMeasurementDirections(gctx);
  row.etaMeasurementDirection = primary;
  if (secondary) {
    row.phiMeasurementDirection = *secondary;
    row.flags |= CudaHitFlags::tiltedStrips;
  }
  return row;
}

/// @brief Row i is the i-th payload; the vector must not be reallocated or
///        reordered while the rows are used
template <typename Payload>
CudaHitPayloadContainer makeCudaHitPayloadContainer(
    const Acts::GeometryContext& gctx, const std::vector<Payload>& hits) {
  CudaHitPayloadContainer container{hits.size()};
  for (std::size_t row = 0u; row < hits.size(); ++row) {
    container.setHit(row, makeCudaHitPayloadRow(gctx, hits[row]));
  }
  return container;
}

}  // namespace ActsExamples
