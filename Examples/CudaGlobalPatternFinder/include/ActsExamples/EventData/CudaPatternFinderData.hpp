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

/// @brief Encode the fields that identify the muon station of a measurement,
///        i.e. the station name, the sector and the detector side. Two hits
///        share a station if their keys are equal, which reproduces
///        MuonSpacePoint::MuonId::sameStation on the device.
/// @param id Identifier of the space point
/// @return Station key of the space point
inline std::uint32_t stationKey(const MuonSpacePoint::MuonId& id) noexcept {
  const auto station = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(Acts::toUnderlying(id.msStation())));
  const auto side = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(Acts::toUnderlying(id.side())));

  return (side << 24u) | (station << 16u) | id.sector();
}

/// @brief Page-locked host memory kept between events, which the device
///        copies to and from without a staging copy and without blocking the
///        host. It grows to the largest request and does not shrink.
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

/// @brief Device memory kept between events. It grows to the largest
///        request and does not shrink; growing waits for the device.
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

/// @brief Columns laid out at the same offsets in a host and a device
///        block, so that one copy moves them all. With null blocks it only
///        counts the bytes.
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

/// @brief Bits of the flags column of the hit payload. They are the yes/no
///        properties of a hit that the pattern finder branches on.
struct CudaHitFlags {
  /// @brief Drift tube measurement, the sensor is the wire
  static constexpr std::uint8_t straw = 1u << 0u;
  /// @brief The hit measures the precision coordinate
  static constexpr std::uint8_t measuresEta = 1u << 1u;
  /// @brief The hit measures the non-precision coordinate as well
  static constexpr std::uint8_t measuresPhi = 1u << 2u;
  /// @brief Precision hit, as HitPayload::isPrecision()
  static constexpr std::uint8_t precision = 1u << 3u;
};

/// @brief One hit payload as plain numbers: what the pattern finder reads from
///        a hit, without the surface or the space point.
struct CudaHitPayloadRow {
  /// @brief Global position of the hit
  Acts::Vector3 position{Acts::Vector3::Zero()};
  /// @brief Global direction of the wire or of the strip
  Acts::Vector3 sensorDirection{Acts::Vector3::Zero()};
  /// @brief Primary measured direction of a strip, the first of
  ///        stripMeasurementDirections. Zero for a straw
  Acts::Vector3 etaMeasurementDirection{Acts::Vector3::Zero()};
  /// @brief Secondary measured direction of a strip that measures phi too,
  ///        the second of stripMeasurementDirections. Zero otherwise
  Acts::Vector3 phiMeasurementDirection{Acts::Vector3::Zero()};

  /// @brief Local y coordinate of the space point, which breaks the ties
  ///        between hits of the same layer and tells measurements apart
  double localPositionY{0.};
  /// @brief Space point covariance of the bending (precision) coordinate
  double etaCovariance{0.};
  /// @brief Space point covariance of the non-bending coordinate
  double phiCovariance{0.};
  /// @brief Straw: variance of the drift circle, drift radius squared plus
  ///        the bending covariance. Zero for a strip
  double driftCovariance{0.};

  /// @brief Geometry identifier of the measurement surface
  std::uint64_t geometryId{0u};
  /// @brief Station key as returned by stationKey()
  std::uint32_t stationKey{0u};
  /// @brief Station index
  std::uint8_t station{0u};
  /// @brief Layer number in the sector frame
  std::uint8_t locLayer{0u};
  /// @brief Bits of CudaHitFlags
  std::uint8_t flags{0u};
};

/// @brief Device-side raw structure-of-arrays view of the pattern finder hit
///        payloads.
///
/// The structure holds raw device pointers and owns no memory. CUDA kernels
/// receive it by value. One row holds the quantities of one hit payload that
/// the pattern finder reads, see CudaHitPayloadRow for what each column is.
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

/// @brief This is the RAM copy of the data. The container copies this data to
/// VRAM with moveToDevice(stream) and copies it back with moveToHost(stream).
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

/// @brief CUDA-backed flat SoA container of the pattern finder hit payloads.
///
/// Row i of the container holds the payload that the search tree refers to as
/// the i-th element of its payload storage, so that a tree entry is translated
/// into a row by pointer arithmetic on that storage. The row order is therefore
/// fixed by the caller filling the container.
class CudaHitPayloadContainer {
 public:
  /// Type alias for container size type.
  using size_type = std::size_t;

  /// Empty default constructor.
  CudaHitPayloadContainer() = default;

  /// Constructor with fixed number of hit payloads.
  /// @param size The number of hit payloads.
  explicit CudaHitPayloadContainer(size_type size);

  /// Deleted because the container owns CUDA device memory.
  CudaHitPayloadContainer(const CudaHitPayloadContainer&) = delete;
  CudaHitPayloadContainer& operator=(const CudaHitPayloadContainer&) = delete;

  /// Movable to transfer ownership of host/device storage.
  CudaHitPayloadContainer(CudaHitPayloadContainer&& other) noexcept;
  CudaHitPayloadContainer& operator=(CudaHitPayloadContainer&& other) noexcept;

  /// @brief Destructor releases VRAM pointers.
  ~CudaHitPayloadContainer() noexcept;

  /// @brief Returns the number of hit payloads.
  size_type size() const noexcept { return m_size; }

  /// @brief Checks whether the container is empty.
  bool empty() const noexcept { return size() == 0; }

  /// @brief Sets every quantity of one hit payload.
  /// @param index The hit payload index.
  /// @param row The quantities of the hit.
  void setHit(size_type index, const CudaHitPayloadRow& row);

  /// @brief Copies all host columns on a CUDA stream.
  /// The method waits only for the supplied stream.
  void moveToDevice(cudaStream_t stream);

  /// @brief Places the device columns of a number of hit payloads that a
  ///        kernel fills in device memory the caller owns, from the device
  ///        address of `block` on, without host columns. The host accessors
  ///        are not usable afterwards. Without a host block only the bytes
  ///        are counted.
  /// @param size The number of hit payloads
  /// @param block The block the columns are placed in
  void placeOnDevice(size_type size, CudaStagedBlock& block);

  /// @brief Releases all device memory.
  void clearDevice() noexcept;

  /// @brief Checks whether device memory is currently allocated.
  bool isOnDevice() const noexcept { return m_onDevice; }

  /// @brief Returns raw device arrays for CUDA kernels.
  CudaHitPayloadArrays deviceArrays() const noexcept { return m_device; }

 private:
  size_type m_size = 0;
  CudaHitPayloadHostData m_host{};
  CudaHitPayloadArrays m_device{};
  /// @brief The one device allocation all columns live in
  void* m_block = nullptr;
  bool m_onDevice = false;

  void checkIndex(size_type index) const;
};

/// @brief Device-side raw structure-of-arrays view of the per-seed candidate
///        lists. It owns no memory; kernels receive it by value. Seed s owns
///        the payload rows [candidateOffsets[s], candidateOffsets[s + 1]) of
///        candidateIndices, which the seed checks sort in place.
struct CudaCandidateArrays {
  std::uint32_t* seedHitIndex = nullptr;
  double* seedSector = nullptr;
  double* seedTheta = nullptr;

  std::uint32_t* candidateOffsets = nullptr;
  std::uint32_t* candidateIndices = nullptr;

  /// @brief Output: CudaSeedStatus of each seed
  std::uint8_t* seedStatus = nullptr;
  /// @brief Output: position of the seed hit in its sorted candidate list
  std::uint16_t* seedPosition = nullptr;
  /// @brief Output: global layer number of each candidate
  std::uint16_t* candidateGlobLayer = nullptr;
  /// @brief Output: the group of each candidate in the low three bits, and
  ///        whether it measures phi (bit 3) and is a precision hit (bit 4)
  std::uint8_t* candidateInfo = nullptr;

  std::uint32_t nSeeds = 0;
  std::uint32_t nCandidates = 0;

  /// @brief First candidate of a seed, as an index into candidateIndices
  __host__ __device__ std::uint32_t begin(std::uint32_t seed) const noexcept {
    return candidateOffsets[seed];
  }
  /// @brief One-past-last candidate of a seed, as an index into
  ///        candidateIndices
  __host__ __device__ std::uint32_t end(std::uint32_t seed) const noexcept {
    return candidateOffsets[seed + 1u];
  }
};

/// @brief Host copy of the candidate lists. The container copies this data to
///        VRAM with moveToDevice(stream) and copies it back with
///        moveToHost(stream).
struct CudaCandidateHostData {
  std::vector<std::uint32_t> seedHitIndex;
  std::vector<double> seedSector;
  std::vector<double> seedTheta;

  /// @brief Start index of each seed in candidateIndices. Size is nSeeds() + 1,
  ///        starting as {0}
  std::vector<std::uint32_t> candidateOffsets{0};
  std::vector<std::uint32_t> candidateIndices;

  std::vector<std::uint8_t> seedStatus;
  std::vector<std::uint16_t> seedPosition;
  std::vector<std::uint16_t> candidateGlobLayer;
  std::vector<std::uint8_t> candidateInfo;
};

/// @brief Values of the seed status column, in the order the CPU seed loop
///        applies its cuts
struct CudaSeedStatus {
  /// @brief The seed preparation has not run
  static constexpr std::uint8_t pending = 0u;
  /// @brief The seed goes on to pattern building
  static constexpr std::uint8_t accepted = 1u;
  /// @brief Fewer candidates than minStripEtaLayers + minPrecisionLayers
  static constexpr std::uint8_t tooFewHits = 2u;
  /// @brief Fewer than minGroups groups with minGroupLayers candidates
  static constexpr std::uint8_t tooFewGroups = 3u;
  /// @brief Fewer layers than minStripEtaLayers + minPrecisionLayers
  static constexpr std::uint8_t tooFewLayers = 4u;
  /// @brief The list does not fit the 16-bit positions; left to the CPU
  static constexpr std::uint8_t overflow = 5u;
};

/// @brief CUDA-backed per-seed candidate lists: a flat list of payload rows
///        and the start index of each seed in that list.
///
/// One seed is one tree entry that passed goodForSeeding. Its candidates are
/// the hits of the range search around that entry, stored as row indices into
/// the payload container. Seeds may share hit rows. The host skip rule
/// (maxSeedAttempts / resolveOverlaps) is left to the driver, so every seed is
/// present.
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

  /// @brief Returns the number of seeds
  size_type nSeeds() const noexcept {
    return m_deviceOnly ? m_device.nSeeds : m_host.seedHitIndex.size();
  }

  /// @brief Returns the total number of candidate entries
  size_type nCandidates() const noexcept {
    return m_deviceOnly ? m_device.nCandidates : m_host.candidateIndices.size();
  }

  /// @brief Returns the number of candidate hits of one seed
  size_type nCandidates(size_type seed) const;

  /// @brief Checks whether there are no seeds
  bool empty() const noexcept { return nSeeds() == 0; }

  /// @brief Appends a seed and the hit rows of its tree range search
  /// @param seedHitIndex Row of the seed hit in the payload container
  /// @param sector Tree sector coordinate of the seed
  /// @param theta Tree theta coordinate of the seed
  /// @param hitIndices Payload rows of the candidate hits
  void addSeed(std::uint32_t seedHitIndex, double sector, double theta,
               std::span<const std::uint32_t> hitIndices);

  /// @brief Overload taking a braced list of hit rows
  void addSeed(std::uint32_t seedHitIndex, double sector, double theta,
               std::initializer_list<std::uint32_t> hitIndices) {
    addSeed(
        seedHitIndex, sector, theta,
        std::span<const std::uint32_t>{hitIndices.begin(), hitIndices.size()});
  }

  /// @brief First candidate of a seed, as an index into the flat list
  size_type candidateBegin(size_type seed) const;

  /// @brief One-past-last candidate of a seed, as an index into the flat list
  size_type candidateEnd(size_type seed) const;

  /// @brief Payload rows of every candidate of a seed
  std::span<const std::uint32_t> candidateHitIndices(size_type seed) const;

  /// @brief CudaSeedStatus of a seed, after the seed preparation
  std::uint8_t seedStatus(size_type seed) const;

  /// @brief Position of the seed hit in its sorted list, after the seed
  ///        preparation
  std::uint16_t seedPosition(size_type seed) const;

  /// @brief Global layer numbers of every candidate of a seed, after the seed
  ///        preparation
  std::span<const std::uint16_t> candidateGlobLayers(size_type seed) const;

  /// @brief Copies the input columns on a CUDA stream and zeroes the output
  ///        columns. The method waits only for the supplied stream.
  void moveToDevice(cudaStream_t stream);

  /// @brief Allocates the device columns of lists that a kernel fills,
  ///        without host columns: nSeeds seeds and nCandidates candidates in
  ///        all, the output columns zeroed. The host accessors other than
  ///        the counts are not usable afterwards.
  void allocateDevice(size_type nSeeds, size_type nCandidates,
                      cudaStream_t stream);

  /// @brief Copies all device columns on a CUDA stream.
  /// The method waits only for the supplied stream.
  void moveToHost(cudaStream_t stream);

  /// @brief Releases all device memory.
  void clearDevice() noexcept;

  /// @brief Checks whether device memory is currently allocated.
  bool isOnDevice() const noexcept { return m_onDevice; }

  /// @brief Returns raw device arrays for CUDA kernels.
  CudaCandidateArrays deviceArrays() const noexcept { return m_device; }

 private:
  CudaCandidateHostData m_host{};
  CudaCandidateArrays m_device{};
  /// @brief The one device allocation all columns live in
  void* m_block = nullptr;
  bool m_onDevice = false;
  /// @brief The lists exist on the device only, with the counts of m_device
  bool m_deviceOnly = false;

  void checkSeed(size_type seed) const;
};

/// @brief Position in a pattern's own hit list that marks an unset hit
///        reference
constexpr std::uint16_t cudaInvalidHitPosition = 0xFFFFu;

/// @brief Bits of the flags column of a pattern record
struct CudaPatternFlags {
  /// @brief The pattern line is drawn from the beamspot rather than two hits
  static constexpr std::uint8_t useBeamspot = 1u << 0u;
  /// @brief The line has to be recomputed when a hit is added on a new layer
  static constexpr std::uint8_t needLineUpdate = 1u << 1u;
  /// @brief The pattern lost against an overlapping one and is dropped
  static constexpr std::uint8_t isOverlap = 1u << 3u;
};

/// @brief Status of a pattern slot
struct CudaPatternStatus {
  /// @brief The slot has not been written
  static constexpr std::uint8_t empty = 0u;
  /// @brief The kernel built a pattern
  static constexpr std::uint8_t ok = 1u;
  /// @brief The pattern failed the quality cuts
  static constexpr std::uint8_t rejected = 2u;
};

/// @brief Bits of the overflow column of the pattern records: what did not fit
struct CudaPatternOverflow {
  /// @brief A seed needed more patterns at once than its slots
  static constexpr std::uint8_t patterns = 1u << 0u;
  /// @brief A pattern needed more hits than its hit slots
  static constexpr std::uint8_t hits = 1u << 1u;
  /// @brief The event had more accepted patterns than its slots
  static constexpr std::uint8_t eventPatterns = 1u << 2u;
};

/// @brief One pattern as plain numbers, the way the host container takes and
///        returns it. This is PatternStateAux without the host pointers.
struct CudaPatternRecordRow {
  /// @brief Expanded sector index of the pattern
  std::int8_t sector{0};
  /// @brief Pattern phi and theta, and the offset of the bending plane from
  ///        the origin along its normal
  double patPhi{0.};
  double patTheta{0.};
  double patPhiOffset{0.};
  /// @brief Covariance of the bending plane's phi and offset:
  ///        PatternStateAux::BendPlaneCov ePhiPhi, eSS and ePhiS
  double bendPlaneCovPhiPhi{0.};
  double bendPlaneCovSS{0.};
  double bendPlaneCovPhiS{0.};
  /// @brief The sum over the pattern's eta hits of the squared residual over
  ///        uncertainty, PatternStateAux::sumNormResidual2
  double sumNormResidual2{0.};
  /// @brief Degrees of freedom of the residuals of the pattern
  std::uint32_t nDoF{0u};
  /// @brief Residual, uncertainty and degrees of freedom of the last inserted
  ///        hit
  double lastResidual{0.};
  double lastResSigma{0.};
  std::uint8_t lastNDoF{0u};
  /// @brief Transverse distance of the line anchor hit from the beam line, the
  ///        radius at which the overlap test compares the phi of two patterns
  double anchorRadius{0.};
  /// @brief Position and direction of the pattern line, in the bending plane
  Acts::Vector3 linePos{Acts::Vector3::Zero()};
  Acts::Vector3 lineDir{Acts::Vector3::Zero()};
  /// @brief Distance between the two points that define the pattern line
  double leverArm{0.};
  /// @brief Normal of the bending plane
  Acts::Vector3 bendPlaneNorm{Acts::Vector3::Zero()};
  /// @brief Counts of precision / trigger / phi layers
  std::uint8_t nPrecisionLayers{0u};
  std::uint8_t nTriggerLayers{0u};
  std::uint8_t nPhiLayers{0u};
  /// @brief Number of hits stored in this pattern's slot
  std::uint32_t nHits{0u};
  /// @brief Positions in this pattern's own hit list of the last inserted hit,
  ///        the last hit of the previous layer and the line anchor. A
  ///        position gives both the payload row and the global layer of the
  ///        hit, which the CPU compares. cudaInvalidHitPosition if unset
  std::uint16_t lastInsertedHit{cudaInvalidHitPosition};
  std::uint16_t prevLayerHit{cudaInvalidHitPosition};
  std::uint16_t lineAnchorHit{cudaInvalidHitPosition};
  /// @brief Bits of CudaPatternFlags
  std::uint8_t flags{0u};
  /// @brief One of CudaPatternStatus
  std::uint8_t status{CudaPatternStatus::empty};
};

/// @brief Device-side raw structure-of-arrays view of the pattern records. It
///        owns no memory; kernels receive it by value. Entry e holds
///        capacityPerSeed pattern slots, nPatterns[e] of them filled, and
///        overflow[e] holds the CudaPatternOverflow bits. Each pattern slot
///        owns maxHitsPerPattern hit slots.
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

  /// @brief Payload row and global layer of every hit slot
  std::uint32_t* hitIndex = nullptr;
  std::uint16_t* globLayer = nullptr;

  /// @brief Number of filled slots and the overflow flag of every seed
  std::uint32_t* nPatterns = nullptr;
  std::uint8_t* overflow = nullptr;

  std::uint32_t nSeeds = 0;
  std::uint32_t capacityPerSeed = 0;
  std::uint32_t maxHitsPerPattern = 0;

  /// @brief Pattern slot k of a seed
  __host__ __device__ std::uint32_t slot(std::uint32_t seed,
                                         std::uint32_t k) const noexcept {
    return seed * capacityPerSeed + k;
  }
  /// @brief Hit slot `local` of a pattern slot
  __host__ __device__ std::uint32_t hitSlot(
      std::uint32_t patternSlot, std::uint32_t local) const noexcept {
    return patternSlot * maxHitsPerPattern + local;
  }
};

/// @brief Host copy of the pattern records
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

/// @brief CUDA-backed pattern records, the output of the pattern building.
///
/// The records are written by the kernel, so moveToDevice() allocates the
/// device buffers and zeroes the counts, the overflow flags and the slot status
/// without uploading anything, as CudaHoughMaximumBatch does. moveToHost()
/// copies every slot back, used or not. Hit rows are indices into
/// CudaHitPayloadContainer.
class CudaPatternRecordContainer {
 public:
  using size_type = std::size_t;

  /// @brief Constructor with the slots of every seed
  /// @param nSeeds The number of seeds
  /// @param capacityPerSeed The number of pattern slots of each seed
  /// @param maxHitsPerPattern The number of hit slots of each pattern
  CudaPatternRecordContainer(size_type nSeeds, size_type capacityPerSeed,
                             size_type maxHitsPerPattern);

  /// @brief Copies the first slots of device records into the host columns
  ///        of this container, as many as it has, from a host copy of the
  ///        device block the source columns live in: column c is read at
  ///        hostCopy + (c - deviceBase). The source must have the same hit
  ///        slots per pattern and at least as many slots and seeds.
  /// @param source The device columns
  /// @param deviceBase The start of the device block
  /// @param hostCopy The host copy of the device block
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
  /// @brief Returns the number of pattern slots of all seeds together
  size_type nSlots() const noexcept { return m_nSeeds * m_capacityPerSeed; }
  bool empty() const noexcept { return nSlots() == 0; }

  /// @brief Returns pattern slot k of a seed
  size_type slot(size_type seed, size_type k) const;

  /// @brief Number of slots a seed filled
  std::uint32_t nPatterns(size_type seed) const;
  void setNPatterns(size_type seed, std::uint32_t nPatterns);

  /// @brief Whether a seed ran out of slots and has to be rebuilt on the CPU
  /// @brief The CudaPatternOverflow bits of a seed, zero if it fitted
  std::uint8_t overflow(size_type seed) const;
  void setOverflow(size_type seed, std::uint8_t overflow);

  /// @brief Returns every quantity of one pattern slot except the hit list
  CudaPatternRecordRow pattern(size_type patternSlot) const;

  /// @brief Returns how many hit slots of a pattern slot are in use
  std::uint32_t nHits(size_type patternSlot) const;

  /// @brief Payload row of one hit of a pattern slot
  std::uint32_t hitIndex(size_type patternSlot, size_type local) const;

  /// @brief Global layer number of one hit of a pattern slot
  std::uint16_t globLayer(size_type patternSlot, size_type local) const;

  /// @brief Payload rows of the used hits of a pattern slot
  std::span<const std::uint32_t> hitIndices(size_type patternSlot) const;

  /// @brief Releases all device memory.
  void clearDevice() noexcept;

  /// @brief Checks whether device memory is currently allocated.
  bool isOnDevice() const noexcept { return m_onDevice; }

  /// @brief Returns raw device arrays for CUDA kernels.
  CudaPatternRecordArrays deviceArrays() const noexcept { return m_device; }

 private:
  CudaPatternRecordContainer() = default;

  size_type m_nSeeds = 0;
  size_type m_capacityPerSeed = 0;
  size_type m_maxHitsPerPattern = 0;
  CudaPatternRecordHostData m_host{};
  CudaPatternRecordArrays m_device{};
  /// @brief The one device allocation all columns live in
  void* m_block = nullptr;
  bool m_onDevice = false;

  size_type hitSlot(size_type patternSlot, size_type local) const;
  void checkSeed(size_type seed) const;
  void checkSlot(size_type patternSlot) const;
  void checkHit(size_type patternSlot, size_type local) const;
};

/// @brief Position of the bending (precision) coordinate in
///        MuonSpacePoint::covariance(), CompSpacePointAuxiliaries::ResidualIdx
constexpr std::size_t cudaHitEtaCovarianceIndex = 1u;
/// @brief Position of the non-bending coordinate in
///        MuonSpacePoint::covariance()
constexpr std::size_t cudaHitPhiCovarianceIndex = 0u;

/// @brief Evaluates on the host what the pattern finder reads from a hit, with
///        the formulas of the CPU hit. Templated so that the CUDA library does
///        not depend on the track finding library.
///
/// @tparam Payload The hit payload type, ActsExamples::HitPayload
/// @param gctx The geometry context
/// @param hit The hit payload
/// @return The quantities of the hit
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

  // A phi-only hit has neither a sensor direction nor a residual variance: it
  // never takes part in the pattern building on the device
  if (!sp.id().measuresEta()) {
    return row;
  }

  row.sensorDirection = hit.globalSensorDirection(gctx);

  if (sp.isStraw()) {
    // PatternStateAux::intrinsicVariance: the drift circle
    row.driftCovariance = Acts::square(sp.driftRadius()) +
                          sp.covariance()[cudaHitEtaCovarianceIndex];
    return row;
  }

  const auto [primary, secondary] = hit.stripMeasurementDirections(gctx);
  row.etaMeasurementDirection = primary;
  if (secondary) {
    row.phiMeasurementDirection = *secondary;
  }
  return row;
}

/// @brief Builds the container of the hit payloads of an event. Row i is the
///        i-th payload of the vector, so that a payload is translated into a
///        row by its position in the vector and back again. The vector must
///        stay as it is, neither reallocated nor reordered, for as long as the
///        rows are used.
/// @param gctx The geometry context
/// @param hits The hit payloads, in the order of the rows
/// @return The container, on the host
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
