// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
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

/// @brief Device-side raw structure-of-arrays view of the pattern finder hit
///        payloads.
///
/// The structure holds raw device pointers and owns no memory. CUDA kernels
/// receive it by value. One row holds the quantities of one hit payload that
/// the seed preparation reads, which are the global position, the local y
/// coordinate used to break ties between hits of the same layer, and the
/// station, station key and layer number defining the layer ordering.
struct CudaHitPayloadArrays {
  double* positionX = nullptr;
  double* positionY = nullptr;
  double* positionZ = nullptr;

  double* localPositionY = nullptr;

  std::uint32_t* stationKey = nullptr;

  std::uint8_t* station = nullptr;
  std::uint8_t* locLayer = nullptr;
};

/// @brief This is the RAM copy of the data. The container copies this data to
/// VRAM with moveToDevice(stream) and copies it back with moveToHost(stream).
struct CudaHitPayloadHostData {
  std::vector<double> positionX;
  std::vector<double> positionY;
  std::vector<double> positionZ;

  std::vector<double> localPositionY;

  std::vector<std::uint32_t> stationKey;

  std::vector<std::uint8_t> station;
  std::vector<std::uint8_t> locLayer;
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

}  // namespace ActsExamples
