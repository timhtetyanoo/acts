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
///  - the intrinsic variance along a direction c, which depends on c and hence
///    cannot be a single number: it needs the measurement directions and the
///    covariances,
///      straw:  driftCovariance * (|c|^2 - (s.c)^2) [+ (s.c)^2 * phiCovariance]
///      strip:  etaCovariance * (e.c)^2 [+ phiCovariance * (p.c)^2]
///    with s the sensor direction, e and p the two measurement directions;
///  - the variance of the hit in phi, the identifier of its measurement
///    surface for the overlap test and the station and layer numbers that the
///    layer ordering is made of.
struct CudaHitPayloadRow {
  /// @brief Global position of the hit
  Acts::Vector3 position{Acts::Vector3::Zero()};
  /// @brief Global direction of the wire or of the strip
  Acts::Vector3 sensorDirection{Acts::Vector3::Zero()};
  /// @brief Direction of the precision measurement of a strip. Zero for a straw
  Acts::Vector3 etaMeasurementDirection{Acts::Vector3::Zero()};
  /// @brief Direction of the second measurement of a strip that measures phi
  ///        too. Zero otherwise
  Acts::Vector3 phiMeasurementDirection{Acts::Vector3::Zero()};

  /// @brief Local y coordinate of the space point, which breaks the ties
  ///        between hits of the same layer and tells measurements apart
  double localPositionY{0.};
  /// @brief Space point covariance of the precision coordinate
  double etaCovariance{0.};
  /// @brief Space point covariance of the non-precision coordinate
  double phiCovariance{0.};
  /// @brief Straw: variance of the drift circle, drift radius squared plus
  ///        the precision covariance. Zero for a strip
  double driftCovariance{0.};
  /// @brief Variance of the hit in phi [rad^2]
  double phiVariance{0.};

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
  double* phiVariance = nullptr;

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
  std::vector<double> phiVariance;

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
  bool m_onDevice = false;

  void checkIndex(size_type index) const;
};

/// @brief Position of the precision coordinate in MuonSpacePoint::covariance()
constexpr std::size_t cudaHitEtaCovarianceIndex = 0u;
/// @brief Position of the non-precision coordinate in
///        MuonSpacePoint::covariance()
constexpr std::size_t cudaHitPhiCovarianceIndex = 1u;

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
  row.phiVariance = hit.phiVariance(gctx);

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
    // The straw keeps the variance of its drift circle in the member that is
    // the strip angle for the strips
    row.driftCovariance = hit.stripAngle;
    return row;
  }

  const auto& surfLinearTrf =
      hit.surface->localToGlobalTransform(gctx).linear();
  if (sp.id().measuresPhi()) {
    row.etaMeasurementDirection =
        hit.nonOrthogonalStrips ? Acts::Vector3{row.sensorDirection.cross(
                                      surfLinearTrf.col(Acts::eZ))}
                                : Acts::Vector3{surfLinearTrf.col(Acts::eX)};
    row.phiMeasurementDirection = surfLinearTrf.col(Acts::eY);
  } else {
    row.etaMeasurementDirection = surfLinearTrf.col(Acts::eX);
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
