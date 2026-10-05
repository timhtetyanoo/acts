// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/EventData/CudaPatternFinderData.hpp"

#include "ActsExamples/Utilities/CudaUtilities.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

/// The host column and the device column of every field of the hit payloads.
/// The allocation, the release and the copies below work on this list, so a
/// new column is added here and in the two structures only.
constexpr auto hitPayloadColumns = [](auto& host, auto& device, auto&& visit) {
  visit(host.positionX, device.positionX);
  visit(host.positionY, device.positionY);
  visit(host.positionZ, device.positionZ);

  visit(host.sensorDirectionX, device.sensorDirectionX);
  visit(host.sensorDirectionY, device.sensorDirectionY);
  visit(host.sensorDirectionZ, device.sensorDirectionZ);

  visit(host.etaMeasurementDirectionX, device.etaMeasurementDirectionX);
  visit(host.etaMeasurementDirectionY, device.etaMeasurementDirectionY);
  visit(host.etaMeasurementDirectionZ, device.etaMeasurementDirectionZ);

  visit(host.phiMeasurementDirectionX, device.phiMeasurementDirectionX);
  visit(host.phiMeasurementDirectionY, device.phiMeasurementDirectionY);
  visit(host.phiMeasurementDirectionZ, device.phiMeasurementDirectionZ);

  visit(host.localPositionY, device.localPositionY);
  visit(host.etaCovariance, device.etaCovariance);
  visit(host.phiCovariance, device.phiCovariance);
  visit(host.driftCovariance, device.driftCovariance);
  visit(host.phiVariance, device.phiVariance);

  visit(host.geometryId, device.geometryId);

  visit(host.stationKey, device.stationKey);

  visit(host.station, device.station);
  visit(host.locLayer, device.locLayer);
  visit(host.flags, device.flags);
};

/// The column functions of CudaUtilities.hpp applied to every column of the
/// list
template <typename Host, typename Device>
void allocateColumns(const Host& host, Device& device) {
  hitPayloadColumns(
      host, device, [](const auto& hostColumn, auto& deviceColumn) {
        ActsExamples::allocateDeviceColumn(deviceColumn, hostColumn.size());
      });
}

template <typename Host, typename Device>
void freeColumns(const Host& host, Device& device) noexcept {
  hitPayloadColumns(host, device,
                    [](const auto& /*hostColumn*/, auto& deviceColumn) {
                      ActsExamples::freeDeviceColumn(deviceColumn);
                    });
}

template <typename Host, typename Device>
void copyColumnsToDevice(const Host& host, const Device& device,
                         cudaStream_t stream) {
  hitPayloadColumns(
      host, device, [stream](const auto& hostColumn, const auto& deviceColumn) {
        ActsExamples::copyColumnToDevice(deviceColumn, hostColumn, stream);
      });
}

template <typename Host, typename Device>
void copyColumnsToHost(Host& host, const Device& device, cudaStream_t stream) {
  hitPayloadColumns(
      host, device, [stream](auto& hostColumn, const auto& deviceColumn) {
        ActsExamples::copyColumnToHost(hostColumn, deviceColumn, stream);
      });
}

}  // namespace

namespace ActsExamples {

CudaHitPayloadContainer::CudaHitPayloadContainer(size_type size)
    : m_size{size} {
  hitPayloadColumns(m_host, m_device,
                    [size](auto& hostColumn, auto& /*deviceColumn*/) {
                      hostColumn.resize(size);
                    });
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

void CudaHitPayloadContainer::setHit(
    size_type index, const Acts::Vector3& position, double localPositionY,
    std::uint32_t stationKey, std::uint8_t station, std::uint8_t locLayer) {
  // The quantities that are not given are reset, so that a row never keeps
  // values of the hit that was there before
  CudaHitPayloadRow row{};
  row.position = position;
  row.localPositionY = localPositionY;
  row.stationKey = stationKey;
  row.station = station;
  row.locLayer = locLayer;

  setHit(index, row);
}

void CudaHitPayloadContainer::setHit(size_type index,
                                     const CudaHitPayloadRow& row) {
  checkIndex(index);

  m_host.positionX[index] = row.position.x();
  m_host.positionY[index] = row.position.y();
  m_host.positionZ[index] = row.position.z();

  m_host.sensorDirectionX[index] = row.sensorDirection.x();
  m_host.sensorDirectionY[index] = row.sensorDirection.y();
  m_host.sensorDirectionZ[index] = row.sensorDirection.z();

  m_host.etaMeasurementDirectionX[index] = row.etaMeasurementDirection.x();
  m_host.etaMeasurementDirectionY[index] = row.etaMeasurementDirection.y();
  m_host.etaMeasurementDirectionZ[index] = row.etaMeasurementDirection.z();

  m_host.phiMeasurementDirectionX[index] = row.phiMeasurementDirection.x();
  m_host.phiMeasurementDirectionY[index] = row.phiMeasurementDirection.y();
  m_host.phiMeasurementDirectionZ[index] = row.phiMeasurementDirection.z();

  m_host.localPositionY[index] = row.localPositionY;
  m_host.etaCovariance[index] = row.etaCovariance;
  m_host.phiCovariance[index] = row.phiCovariance;
  m_host.driftCovariance[index] = row.driftCovariance;
  m_host.phiVariance[index] = row.phiVariance;

  m_host.geometryId[index] = row.geometryId;

  m_host.stationKey[index] = row.stationKey;

  m_host.station[index] = row.station;
  m_host.locLayer[index] = row.locLayer;
  m_host.flags[index] = row.flags;
}

CudaHitPayloadRow CudaHitPayloadContainer::hit(size_type index) const {
  checkIndex(index);

  CudaHitPayloadRow row{};
  row.position = {m_host.positionX[index], m_host.positionY[index],
                  m_host.positionZ[index]};
  row.sensorDirection = {m_host.sensorDirectionX[index],
                         m_host.sensorDirectionY[index],
                         m_host.sensorDirectionZ[index]};
  row.etaMeasurementDirection = {m_host.etaMeasurementDirectionX[index],
                                 m_host.etaMeasurementDirectionY[index],
                                 m_host.etaMeasurementDirectionZ[index]};
  row.phiMeasurementDirection = {m_host.phiMeasurementDirectionX[index],
                                 m_host.phiMeasurementDirectionY[index],
                                 m_host.phiMeasurementDirectionZ[index]};

  row.localPositionY = m_host.localPositionY[index];
  row.etaCovariance = m_host.etaCovariance[index];
  row.phiCovariance = m_host.phiCovariance[index];
  row.driftCovariance = m_host.driftCovariance[index];
  row.phiVariance = m_host.phiVariance[index];

  row.geometryId = m_host.geometryId[index];
  row.stationKey = m_host.stationKey[index];
  row.station = m_host.station[index];
  row.locLayer = m_host.locLayer[index];
  row.flags = m_host.flags[index];

  return row;
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

  allocateColumns(m_host, m_device);
  copyColumnsToDevice(m_host, m_device, stream);

  m_onDevice = true;
}

void CudaHitPayloadContainer::moveToHost(cudaStream_t stream) {
  if (!m_onDevice) {
    return;
  }

  copyColumnsToHost(m_host, m_device, stream);
}

void CudaHitPayloadContainer::clearDevice() noexcept {
  freeColumns(m_host, m_device);

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
