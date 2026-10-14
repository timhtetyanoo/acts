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

/// @brief One hit payload as plain numbers, the way the host container takes and
///        returns it.
///
/// The values are what the pattern finder reads from a hit through the
/// GlobPatFinderHit interface, evaluated once on the host. A kernel cannot call
/// the surface or the space point, so everything that depends on them is here:
///
///  - the position and the sensor direction;
///  - what PatternStateAux::intrinsicVariance reads to contract the hit
///    covariance along a direction: the space point covariance, the drift
///    variance of a straw and the measured directions of a strip
///    (stripMeasurementDirections);
///  - the identifier of its measurement surface for the overlap test and the
///    station and layer numbers that the layer ordering is made of.
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

  /// @brief Sets the quantities of one hit payload.
  /// @param index The hit payload index.
  /// @param position The global position of the hit.
  /// @param localPositionY The local y coordinate of the space point.
  /// @param stationKey The station key as returned by stationKey().
  /// @param station The station index of the hit.
  /// @param locLayer The layer number in the sector frame.
  void setHit(size_type index, const Acts::Vector3& position,
              double localPositionY, std::uint32_t stationKey,
              std::uint8_t station, std::uint8_t locLayer);

  /// @brief Sets every quantity of one hit payload.
  /// @param index The hit payload index.
  /// @param row The quantities of the hit.
  void setHit(size_type index, const CudaHitPayloadRow& row);

  /// @brief Returns every quantity of one hit payload.
  /// @param index The hit payload index.
  CudaHitPayloadRow hit(size_type index) const;

  /// @brief Returns the global position of a hit payload.
  /// @param index The hit payload index.
  Acts::Vector3 position(size_type index) const;

  /// @brief Returns the local y coordinate of a hit payload.
  /// @param index The hit payload index.
  double localPositionY(size_type index) const;

  /// @brief Returns the station key of a hit payload.
  /// @param index The hit payload index.
  std::uint32_t stationKey(size_type index) const;

  /// @brief Returns the station index of a hit payload.
  /// @param index The hit payload index.
  std::uint8_t station(size_type index) const;

  /// @brief Returns the layer number in the sector frame of a hit payload.
  /// @param index The hit payload index.
  std::uint8_t locLayer(size_type index) const;

  /// @brief Copies all host columns on a CUDA stream.
  /// The method waits only for the supplied stream.
  void moveToDevice(cudaStream_t stream);

  /// @brief Allocates the device columns of a number of hit payloads that a
  ///        kernel fills, without host columns. The host accessors are not
  ///        usable afterwards.
  /// @param size The number of hit payloads
  /// @param stream The stream the allocation is ordered on
  void allocateDevice(size_type size, cudaStream_t stream);

  /// @brief Copies all device columns on a CUDA stream.
  /// The method waits only for the supplied stream.
  void moveToHost(cudaStream_t stream);

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
///        lists. The structure holds raw device pointers and owns no memory.
///        CUDA kernels receive it by value.
///
/// Seed s owns the candidate hit rows
/// `[candidateOffsets[s], candidateOffsets[s + 1])` of candidateIndices.
/// Those rows are indices into CudaHitPayloadContainer. The lists are the
/// output of the host tree range search; the skip rule is not applied. The
/// seed preparation kernel sorts each list in place and fills the output
/// columns.
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
  size_type nSeeds() const noexcept { return m_host.seedHitIndex.size(); }

  /// @brief Returns the total number of candidate entries
  size_type nCandidates() const noexcept {
    return m_host.candidateIndices.size();
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

  /// @brief Overwrites the header of an existing seed. The candidate range
  ///        is left as it is
  void setSeed(size_type seed, std::uint32_t seedHitIndex, double sector,
               double theta);

  /// @brief Overwrites one flat candidate entry
  void setCandidate(size_type index, std::uint32_t hitIndex);

  /// @brief Payload row of the seed hit
  std::uint32_t seedHitIndex(size_type seed) const;

  /// @brief Tree sector coordinate of the seed
  double seedSector(size_type seed) const;

  /// @brief Tree theta coordinate of the seed
  double seedTheta(size_type seed) const;

  /// @brief First candidate of a seed, as an index into the flat list
  size_type candidateBegin(size_type seed) const;

  /// @brief One-past-last candidate of a seed, as an index into the flat list
  size_type candidateEnd(size_type seed) const;

  /// @brief Payload row of one candidate of a seed
  std::uint32_t candidateHitIndex(size_type seed, size_type local) const;

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

  void checkSeed(size_type seed) const;
  void checkCandidate(size_type index) const;
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
  /// @brief The pattern has passed the quality cuts
  static constexpr std::uint8_t isFinalized = 1u << 2u;
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
  /// @brief While the pattern is built, the sum over its eta hits of the
  ///        squared residual over uncertainty. Divided by the number of
  ///        degrees of freedom once the pattern is finalized, as on the CPU
  double meanNormResidual2{0.};
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

/// @brief Device-side raw structure-of-arrays view of the pattern records.
///        The structure holds raw device pointers and owns no memory. CUDA
///        kernels receive it by value.
///
/// Every seed owns capacityPerSeed pattern slots, in order:
///
///   seed 0: slot 0, slot 1, ..., slot capacityPerSeed - 1
///   seed 1: slot 0, ...
///
/// so a seed writes only its own slots and the output order does not depend on
/// the order the threads run in. nPatterns[seed] is how many of them the seed
/// filled, and overflow[seed] holds the CudaPatternOverflow bits of what did
/// not fit; such a seed is built again with more room. Each pattern slot owns
/// maxHitsPerPattern hit slots.
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

  double* meanNormResidual2 = nullptr;
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

  std::vector<double> meanNormResidual2;
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

  /// @brief Records that live on the device only, for the kernels to fill:
  ///        the device columns are allocated on the stream and the counts,
  ///        the overflow flags and the slot status zeroed. There are no host
  ///        columns.
  static CudaPatternRecordContainer onDevice(size_type nSeeds,
                                             size_type capacityPerSeed,
                                             size_type maxHitsPerPattern,
                                             cudaStream_t stream);

  /// @brief Copies the first slots of device records into the host columns
  ///        of this container, as many as it has: the slot columns, their hit
  ///        slots and the per-seed columns. The source must have the same hit
  ///        slots per pattern and at least as many slots and seeds.
  /// The method waits only for the supplied stream.
  void copyFromDevice(const CudaPatternRecordArrays& source,
                      cudaStream_t stream);

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

  /// @brief Sets every quantity of one pattern slot except the hit list
  void setPattern(size_type patternSlot, const CudaPatternRecordRow& row);

  /// @brief Returns every quantity of one pattern slot except the hit list
  CudaPatternRecordRow pattern(size_type patternSlot) const;

  /// @brief Sets one hit of a pattern slot
  /// @param patternSlot The pattern slot
  /// @param local The hit slot inside the pattern
  /// @param hitIndex The payload row of the hit
  /// @param globLayer The global layer number of the hit
  void setHit(size_type patternSlot, size_type local, std::uint32_t hitIndex,
              std::uint16_t globLayer = 0u);

  /// @brief Sets how many hit slots of a pattern slot are in use
  void setNHits(size_type patternSlot, std::uint32_t nHits);

  /// @brief Returns how many hit slots of a pattern slot are in use
  std::uint32_t nHits(size_type patternSlot) const;

  /// @brief Payload row of one hit of a pattern slot
  std::uint32_t hitIndex(size_type patternSlot, size_type local) const;

  /// @brief Global layer number of one hit of a pattern slot
  std::uint16_t globLayer(size_type patternSlot, size_type local) const;

  /// @brief Payload rows of the used hits of a pattern slot
  std::span<const std::uint32_t> hitIndices(size_type patternSlot) const;

  /// @brief Allocates the device buffers and zeroes the counts, the overflow
  ///        flags and the slot status. Nothing is uploaded.
  void moveToDevice(cudaStream_t stream);

  /// @brief Copies the counts, the overflow flags and every slot to the host.
  /// The method waits only for the supplied stream.
  void moveToHost(cudaStream_t stream);

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

/// @brief Evaluates, on the host, everything the pattern finder reads from a
///        hit, so that a kernel needs neither the surface nor the space point.
///
/// The function takes the hit type as a template parameter and reads the public
/// members of ActsExamples::HitPayload, so that the CUDA library has no
/// dependency on the track finding library. Each quantity is computed with the
/// formula of the CPU hit, in the same order of operations, because the device
/// has to reproduce the CPU residuals.
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

/// @brief Builds the per-seed candidate lists of an event from the host tree,
///        as the CPU seed loop does.
///
/// Every tree entry is a seed candidate, in the order of the tree. An entry
/// that is not good for seeding is left out; every other gets the hits of the
/// same range search the CPU makes, in the order the search returns them. A
/// hit enters the tree once per sector it is projected into, so the same hit
/// can be several seeds. The skip rule is not applied: it depends on the
/// patterns found before a seed and is replayed by the driver.
///
/// @tparam Finder The pattern finder, for its tree and coordinate types
///         (ActsExamples::GlobalPatternFinder_t)
/// @param tree The search tree. Its entries point into payloads
/// @param payloads The hit payloads, in the order of the payload container rows
/// @param seedSelector The seed selector of the pattern finder
/// @return The candidate lists, on the host
template <typename Finder, typename Payload, typename Selector>
CudaCandidateListContainer makeCudaCandidateLists(
    const typename Finder::SearchTree_t& tree,
    const std::vector<Payload>& payloads, const Selector& seedSelector) {
  using HitCoords = typename Finder::HitCoords;
  constexpr auto sectorIdx = Acts::toUnderlying(HitCoords::eSector);
  constexpr auto thetaIdx = Acts::toUnderlying(HitCoords::eTheta);
  auto row = [&payloads](const Payload* hit) {
    return static_cast<std::uint32_t>(hit - payloads.data());
  };

  CudaCandidateListContainer lists{};
  std::vector<std::uint32_t> candidates{};
  for (const auto& [seedCoords, seedPtr] : tree) {
    if (!seedSelector.goodForSeeding(*seedPtr)) {
      continue;
    }
    // The window of GlobalPatternFinder::findPatternsInEta
    typename Finder::SearchTree_t::range_t selectRange{};
    selectRange[sectorIdx].shrink(seedCoords[sectorIdx] - 0.1,
                                  seedCoords[sectorIdx] + 0.1);
    const double thetaHalfWindow{seedSelector.thetaSearchWindow(*seedPtr) / 2.};
    selectRange[thetaIdx].shrink(seedCoords[thetaIdx] - thetaHalfWindow,
                                 seedCoords[thetaIdx] + thetaHalfWindow);

    candidates.clear();
    tree.rangeSearchMapDiscard(selectRange,
                               [&](const auto& /*coords*/, const Payload* hit) {
                                 candidates.push_back(row(hit));
                               });
    lists.addSeed(row(seedPtr), seedCoords[sectorIdx], seedCoords[thetaIdx],
                  candidates);
  }
  return lists;
}

}  // namespace ActsExamples
