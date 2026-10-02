// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/EventData/CudaHitPayload.hpp"

#include "ActsExamples/Utilities/CudaUtilities.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

void allocateDeviceData(ActsExamples::CudaHitPayloadArrays& device,
                        std::size_t hitCount) {
  using ActsExamples::allocateDeviceColumn;

  allocateDeviceColumn(device.positionX, hitCount);
  allocateDeviceColumn(device.positionY, hitCount);
  allocateDeviceColumn(device.positionZ, hitCount);

  allocateDeviceColumn(device.localPositionY, hitCount);

  allocateDeviceColumn(device.stationKey, hitCount);

  allocateDeviceColumn(device.station, hitCount);
  allocateDeviceColumn(device.locLayer, hitCount);
}

void freeDeviceData(ActsExamples::CudaHitPayloadArrays& device) noexcept {
  using ActsExamples::freeDeviceColumn;

  freeDeviceColumn(device.positionX);
  freeDeviceColumn(device.positionY);
  freeDeviceColumn(device.positionZ);

  freeDeviceColumn(device.localPositionY);

  freeDeviceColumn(device.stationKey);

  freeDeviceColumn(device.station);
  freeDeviceColumn(device.locLayer);
}

void copyHostToDevice(ActsExamples::CudaHitPayloadArrays& device,
                      const ActsExamples::CudaHitPayloadHostData& host,
                      cudaStream_t stream) {
  using ActsExamples::copyColumnToDevice;

  copyColumnToDevice(device.positionX, host.positionX, stream);
  copyColumnToDevice(device.positionY, host.positionY, stream);
  copyColumnToDevice(device.positionZ, host.positionZ, stream);

  copyColumnToDevice(device.localPositionY, host.localPositionY, stream);

  copyColumnToDevice(device.stationKey, host.stationKey, stream);

  copyColumnToDevice(device.station, host.station, stream);
  copyColumnToDevice(device.locLayer, host.locLayer, stream);
}

void copyDeviceToHost(ActsExamples::CudaHitPayloadHostData& host,
                      const ActsExamples::CudaHitPayloadArrays& device,
                      cudaStream_t stream) {
  using ActsExamples::copyColumnToHost;

  copyColumnToHost(host.positionX, device.positionX, stream);
  copyColumnToHost(host.positionY, device.positionY, stream);
  copyColumnToHost(host.positionZ, device.positionZ, stream);

  copyColumnToHost(host.localPositionY, device.localPositionY, stream);

  copyColumnToHost(host.stationKey, device.stationKey, stream);

  copyColumnToHost(host.station, device.station, stream);
  copyColumnToHost(host.locLayer, device.locLayer, stream);
}

}  // namespace

namespace ActsExamples {

CudaHitPayloadContainer::CudaHitPayloadContainer(size_type size)
    : m_size{size} {
  m_host.positionX.resize(size, 0.);
  m_host.positionY.resize(size, 0.);
  m_host.positionZ.resize(size, 0.);

  m_host.localPositionY.resize(size, 0.);

  m_host.stationKey.resize(size, 0u);

  m_host.station.resize(size, 0u);
  m_host.locLayer.resize(size, 0u);
}

CudaHitPayloadContainer::CudaHitPayloadContainer(
    CudaHitPayloadContainer&& other) noexcept
    : m_size{std::exchange(other.m_size, 0)},
      m_host{std::move(other.m_host)},
      m_device{std::exchange(other.m_device, {})},
      m_onDevice{std::exchange(other.m_onDevice, false)} {}

CudaHitPayloadContainer& CudaHitPayloadContainer::operator=(
    CudaHitPayloadContainer&& other) noexcept {
  if (this != &other) {
    clearDevice();

    m_size = std::exchange(other.m_size, 0);
    m_host = std::move(other.m_host);
    m_device = std::exchange(other.m_device, {});
    m_onDevice = std::exchange(other.m_onDevice, false);
  }

  return *this;
}

CudaHitPayloadContainer::~CudaHitPayloadContainer() noexcept {
  clearDevice();
}

void CudaHitPayloadContainer::setHit(size_type index,
                                     const Acts::Vector3& position,
                                     double localPositionY,
                                     std::uint32_t stationKey,
                                     std::uint8_t station,
                                     std::uint8_t locLayer) {
  checkIndex(index);

  m_host.positionX[index] = position.x();
  m_host.positionY[index] = position.y();
  m_host.positionZ[index] = position.z();

  m_host.localPositionY[index] = localPositionY;

  m_host.stationKey[index] = stationKey;

  m_host.station[index] = station;
  m_host.locLayer[index] = locLayer;
}

Acts::Vector3 CudaHitPayloadContainer::position(size_type index) const {
  checkIndex(index);

  return Acts::Vector3{m_host.positionX[index], m_host.positionY[index],
                       m_host.positionZ[index]};
}

double CudaHitPayloadContainer::localPositionY(size_type index) const {
  checkIndex(index);

  return m_host.localPositionY[index];
}

std::uint32_t CudaHitPayloadContainer::stationKey(size_type index) const {
  checkIndex(index);

  return m_host.stationKey[index];
}

std::uint8_t CudaHitPayloadContainer::station(size_type index) const {
  checkIndex(index);

  return m_host.station[index];
}

std::uint8_t CudaHitPayloadContainer::locLayer(size_type index) const {
  checkIndex(index);

  return m_host.locLayer[index];
}

void CudaHitPayloadContainer::moveToDevice(cudaStream_t stream) {
  clearDevice();

  allocateDeviceData(m_device, m_size);
  copyHostToDevice(m_device, m_host, stream);

  m_onDevice = true;
}

void CudaHitPayloadContainer::moveToHost(cudaStream_t stream) {
  if (!m_onDevice) {
    return;
  }

  copyDeviceToHost(m_host, m_device, stream);
}

void CudaHitPayloadContainer::clearDevice() noexcept {
  freeDeviceData(m_device);

  m_device = {};
  m_onDevice = false;
}

void CudaHitPayloadContainer::checkIndex(size_type index) const {
  if (index >= m_size) {
    std::stringstream ss;
    ss << "CudaHitPayloadContainer: index " << index << " is out of range for "
       << m_size << " hit payloads";
    throw std::out_of_range(ss.str());
  }
}

}  // namespace ActsExamples
