// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/EventData/CudaPatternFinderData.hpp"

#include "ActsExamples/Utilities/CudaUtilities.hpp"

#include <span>
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

/// The host column and the device column of every input field of the
/// candidate lists. Offsets are one longer than the per-seed columns; the visit
/// uses each vector's own size.
constexpr auto candidateInputColumns = [](auto& host, auto& device,
                                          auto&& visit) {
  visit(host.seedHitIndex, device.seedHitIndex);
  visit(host.seedSector, device.seedSector);
  visit(host.seedTheta, device.seedTheta);
  visit(host.candidateOffsets, device.candidateOffsets);
  visit(host.candidateIndices, device.candidateIndices);
};

/// The columns the seed preparation kernel writes
constexpr auto candidateOutputColumns = [](auto& host, auto& device,
                                           auto&& visit) {
  visit(host.seedStatus, device.seedStatus);
  visit(host.seedPosition, device.seedPosition);
  visit(host.candidateGlobLayer, device.candidateGlobLayer);
};

constexpr auto candidateColumns = [](auto& host, auto& device, auto&& visit) {
  candidateInputColumns(host, device, visit);
  candidateOutputColumns(host, device, visit);
};

/// The host column and the device column of every per-slot field of the
/// pattern records
constexpr auto patternSlotColumns = [](auto& host, auto& device, auto&& visit) {
  visit(host.sector, device.sector);

  visit(host.patPhi, device.patPhi);
  visit(host.patTheta, device.patTheta);
  visit(host.patPhiCov, device.patPhiCov);

  visit(host.meanNormResidual2, device.meanNormResidual2);
  visit(host.lastResidual, device.lastResidual);
  visit(host.lastResSigma, device.lastResSigma);

  visit(host.linePosX, device.linePosX);
  visit(host.linePosY, device.linePosY);
  visit(host.linePosZ, device.linePosZ);

  visit(host.lineDirX, device.lineDirX);
  visit(host.lineDirY, device.lineDirY);
  visit(host.lineDirZ, device.lineDirZ);

  visit(host.leverArm, device.leverArm);

  visit(host.bendPlaneNormX, device.bendPlaneNormX);
  visit(host.bendPlaneNormY, device.bendPlaneNormY);
  visit(host.bendPlaneNormZ, device.bendPlaneNormZ);

  visit(host.nPrecisionLayers, device.nPrecisionLayers);
  visit(host.nTriggerLayers, device.nTriggerLayers);
  visit(host.nPhiLayers, device.nPhiLayers);

  visit(host.nHits, device.nHits);

  visit(host.lastInsertedHit, device.lastInsertedHit);
  visit(host.prevLayerHit, device.prevLayerHit);
  visit(host.lineAnchorHit, device.lineAnchorHit);

  visit(host.flags, device.flags);
  visit(host.status, device.status);
};

/// Per hit slot
constexpr auto patternHitColumns = [](auto& host, auto& device, auto&& visit) {
  visit(host.hitIndex, device.hitIndex);
  visit(host.globLayer, device.globLayer);
};

/// Per seed
constexpr auto patternSeedColumns = [](auto& host, auto& device, auto&& visit) {
  visit(host.nPatterns, device.nPatterns);
  visit(host.overflow, device.overflow);
};

constexpr auto patternColumns = [](auto& host, auto& device, auto&& visit) {
  patternSlotColumns(host, device, visit);
  patternHitColumns(host, device, visit);
  patternSeedColumns(host, device, visit);
};

/// Sets a device column to zero on a stream
template <typename T>
void zeroDeviceColumn(T* deviceColumn, std::size_t size, cudaStream_t stream) {
  if (size == 0 || deviceColumn == nullptr) {
    return;
  }
  ACTS_CUDA_CHECK(cudaMemsetAsync(deviceColumn, 0, size * sizeof(T), stream));
}

/// The column functions of CudaUtilities.hpp applied to every column of a
/// list
template <typename Columns, typename Host, typename Device>
void allocateColumns(Columns&& columns, const Host& host, Device& device) {
  columns(host, device, [](const auto& hostColumn, auto& deviceColumn) {
    ActsExamples::allocateDeviceColumn(deviceColumn, hostColumn.size());
  });
}

template <typename Columns, typename Host, typename Device>
void freeColumns(Columns&& columns, const Host& host, Device& device) noexcept {
  columns(host, device, [](const auto& /*hostColumn*/, auto& deviceColumn) {
    ActsExamples::freeDeviceColumn(deviceColumn);
  });
}

template <typename Columns, typename Host, typename Device>
void copyColumnsToDevice(Columns&& columns, const Host& host,
                         const Device& device, cudaStream_t stream) {
  columns(host, device,
          [stream](const auto& hostColumn, const auto& deviceColumn) {
            ActsExamples::copyColumnToDevice(deviceColumn, hostColumn, stream);
          });
}

template <typename Columns, typename Host, typename Device>
void copyColumnsToHost(Columns&& columns, Host& host, const Device& device,
                       cudaStream_t stream) {
  columns(host, device, [stream](auto& hostColumn, const auto& deviceColumn) {
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

  allocateColumns(hitPayloadColumns, m_host, m_device);
  copyColumnsToDevice(hitPayloadColumns, m_host, m_device, stream);

  m_onDevice = true;
}

void CudaHitPayloadContainer::moveToHost(cudaStream_t stream) {
  if (!m_onDevice) {
    return;
  }

  copyColumnsToHost(hitPayloadColumns, m_host, m_device, stream);
}

void CudaHitPayloadContainer::clearDevice() noexcept {
  freeColumns(hitPayloadColumns, m_host, m_device);

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

CudaCandidateListContainer::CudaCandidateListContainer(
    CudaCandidateListContainer&& other) noexcept
    : m_host{std::move(other.m_host)},
      m_device{std::exchange(other.m_device, {})},
      m_onDevice{std::exchange(other.m_onDevice, false)} {
  other.m_host = {};
}

CudaCandidateListContainer& CudaCandidateListContainer::operator=(
    CudaCandidateListContainer&& other) noexcept {
  if (this != &other) {
    clearDevice();

    m_host = std::move(other.m_host);
    m_device = std::exchange(other.m_device, {});
    m_onDevice = std::exchange(other.m_onDevice, false);
    other.m_host = {};
  }

  return *this;
}

CudaCandidateListContainer::~CudaCandidateListContainer() noexcept {
  clearDevice();
}

CudaCandidateListContainer::size_type CudaCandidateListContainer::nCandidates(
    size_type seed) const {
  return candidateEnd(seed) - candidateBegin(seed);
}

void CudaCandidateListContainer::addSeed(
    std::uint32_t seedHitIndex, double sector, double theta,
    std::span<const std::uint32_t> hitIndices) {
  // Growing the CSR invalidates the device buffers
  clearDevice();

  m_host.seedHitIndex.push_back(seedHitIndex);
  m_host.seedSector.push_back(sector);
  m_host.seedTheta.push_back(theta);
  m_host.candidateIndices.insert(m_host.candidateIndices.end(),
                                 hitIndices.begin(), hitIndices.end());
  m_host.candidateOffsets.push_back(
      static_cast<std::uint32_t>(m_host.candidateIndices.size()));

  m_host.seedStatus.push_back(CudaSeedStatus::pending);
  m_host.seedPosition.push_back(0u);
  m_host.candidateGlobLayer.resize(m_host.candidateIndices.size(), 0u);
}

void CudaCandidateListContainer::setSeed(size_type seed,
                                         std::uint32_t seedHitIndex,
                                         double sector, double theta) {
  checkSeed(seed);

  m_host.seedHitIndex[seed] = seedHitIndex;
  m_host.seedSector[seed] = sector;
  m_host.seedTheta[seed] = theta;
}

void CudaCandidateListContainer::setCandidate(size_type index,
                                              std::uint32_t hitIndex) {
  checkCandidate(index);

  m_host.candidateIndices[index] = hitIndex;
}

std::uint32_t CudaCandidateListContainer::seedHitIndex(size_type seed) const {
  checkSeed(seed);

  return m_host.seedHitIndex[seed];
}

double CudaCandidateListContainer::seedSector(size_type seed) const {
  checkSeed(seed);

  return m_host.seedSector[seed];
}

double CudaCandidateListContainer::seedTheta(size_type seed) const {
  checkSeed(seed);

  return m_host.seedTheta[seed];
}

CudaCandidateListContainer::size_type
CudaCandidateListContainer::candidateBegin(size_type seed) const {
  checkSeed(seed);

  return m_host.candidateOffsets[seed];
}

CudaCandidateListContainer::size_type CudaCandidateListContainer::candidateEnd(
    size_type seed) const {
  checkSeed(seed);

  return m_host.candidateOffsets[seed + 1u];
}

std::uint32_t CudaCandidateListContainer::candidateHitIndex(
    size_type seed, size_type local) const {
  const size_type begin{candidateBegin(seed)};
  const size_type index{begin + local};
  if (index >= candidateEnd(seed)) {
    std::stringstream ss;
    ss << "CudaCandidateListContainer: local candidate " << local
       << " is out of range for seed " << seed << " with " << nCandidates(seed)
       << " candidates";
    throw std::out_of_range(ss.str());
  }

  return m_host.candidateIndices[index];
}

std::span<const std::uint32_t> CudaCandidateListContainer::candidateHitIndices(
    size_type seed) const {
  const size_type begin{candidateBegin(seed)};
  const size_type end{candidateEnd(seed)};
  return std::span<const std::uint32_t>{m_host.candidateIndices.data() + begin,
                                        end - begin};
}

std::uint8_t CudaCandidateListContainer::seedStatus(size_type seed) const {
  checkSeed(seed);

  return m_host.seedStatus[seed];
}

std::uint16_t CudaCandidateListContainer::seedPosition(size_type seed) const {
  checkSeed(seed);

  return m_host.seedPosition[seed];
}

std::span<const std::uint16_t> CudaCandidateListContainer::candidateGlobLayers(
    size_type seed) const {
  const size_type begin{candidateBegin(seed)};
  const size_type end{candidateEnd(seed)};
  return std::span<const std::uint16_t>{
      m_host.candidateGlobLayer.data() + begin, end - begin};
}

void CudaCandidateListContainer::moveToDevice(cudaStream_t stream) {
  clearDevice();

  allocateColumns(candidateColumns, m_host, m_device);
  copyColumnsToDevice(candidateInputColumns, m_host, m_device, stream);
  // Zeroed, so that a seed the kernel has not reached reads as pending
  candidateOutputColumns(
      m_host, m_device, [stream](const auto& hostColumn, auto& deviceColumn) {
        zeroDeviceColumn(deviceColumn, hostColumn.size(), stream);
      });

  m_device.nSeeds = static_cast<std::uint32_t>(nSeeds());
  m_device.nCandidates = static_cast<std::uint32_t>(nCandidates());
  m_onDevice = true;
}

void CudaCandidateListContainer::moveToHost(cudaStream_t stream) {
  if (!m_onDevice) {
    return;
  }

  copyColumnsToHost(candidateColumns, m_host, m_device, stream);
}

void CudaCandidateListContainer::clearDevice() noexcept {
  freeColumns(candidateColumns, m_host, m_device);

  m_device = {};
  m_onDevice = false;
}

void CudaCandidateListContainer::checkSeed(size_type seed) const {
  if (seed >= nSeeds()) {
    std::stringstream ss;
    ss << "CudaCandidateListContainer: seed " << seed << " is out of range for "
       << nSeeds() << " seeds";
    throw std::out_of_range(ss.str());
  }
}

void CudaCandidateListContainer::checkCandidate(size_type index) const {
  if (index >= nCandidates()) {
    std::stringstream ss;
    ss << "CudaCandidateListContainer: candidate " << index
       << " is out of range for " << nCandidates() << " candidates";
    throw std::out_of_range(ss.str());
  }
}

CudaPatternRecordContainer::CudaPatternRecordContainer(
    size_type nSeeds, size_type capacityPerSeed, size_type maxHitsPerPattern)
    : m_nSeeds{nSeeds},
      m_capacityPerSeed{capacityPerSeed},
      m_maxHitsPerPattern{maxHitsPerPattern} {
  if (maxHitsPerPattern >= cudaInvalidHitPosition) {
    throw std::invalid_argument(
        "CudaPatternRecordContainer: maxHitsPerPattern does not fit the hit "
        "positions");
  }
  const size_type slots{nSlots()};
  const size_type hitSlots{slots * maxHitsPerPattern};

  patternSlotColumns(m_host, m_device,
                     [slots](auto& hostColumn, auto& /*deviceColumn*/) {
                       hostColumn.resize(slots);
                     });
  patternHitColumns(m_host, m_device,
                    [hitSlots](auto& hostColumn, auto& /*deviceColumn*/) {
                      hostColumn.resize(hitSlots);
                    });
  patternSeedColumns(m_host, m_device,
                     [nSeeds](auto& hostColumn, auto& /*deviceColumn*/) {
                       hostColumn.resize(nSeeds);
                     });

  m_host.lastInsertedHit.assign(slots, cudaInvalidHitPosition);
  m_host.prevLayerHit.assign(slots, cudaInvalidHitPosition);
  m_host.lineAnchorHit.assign(slots, cudaInvalidHitPosition);
}

CudaPatternRecordContainer::CudaPatternRecordContainer(
    CudaPatternRecordContainer&& other) noexcept
    : m_nSeeds{std::exchange(other.m_nSeeds, 0)},
      m_capacityPerSeed{std::exchange(other.m_capacityPerSeed, 0)},
      m_maxHitsPerPattern{std::exchange(other.m_maxHitsPerPattern, 0)},
      m_host{std::move(other.m_host)},
      m_device{std::exchange(other.m_device, {})},
      m_onDevice{std::exchange(other.m_onDevice, false)} {}

CudaPatternRecordContainer& CudaPatternRecordContainer::operator=(
    CudaPatternRecordContainer&& other) noexcept {
  if (this != &other) {
    clearDevice();

    m_nSeeds = std::exchange(other.m_nSeeds, 0);
    m_capacityPerSeed = std::exchange(other.m_capacityPerSeed, 0);
    m_maxHitsPerPattern = std::exchange(other.m_maxHitsPerPattern, 0);
    m_host = std::move(other.m_host);
    m_device = std::exchange(other.m_device, {});
    m_onDevice = std::exchange(other.m_onDevice, false);
  }

  return *this;
}

CudaPatternRecordContainer::~CudaPatternRecordContainer() noexcept {
  clearDevice();
}

CudaPatternRecordContainer::size_type CudaPatternRecordContainer::slot(
    size_type seed, size_type k) const {
  checkSeed(seed);
  if (k >= m_capacityPerSeed) {
    std::stringstream ss;
    ss << "CudaPatternRecordContainer: slot " << k
       << " is out of range for capacityPerSeed " << m_capacityPerSeed;
    throw std::out_of_range(ss.str());
  }

  return seed * m_capacityPerSeed + k;
}

std::uint32_t CudaPatternRecordContainer::nPatterns(size_type seed) const {
  checkSeed(seed);

  return m_host.nPatterns[seed];
}

void CudaPatternRecordContainer::setNPatterns(size_type seed,
                                              std::uint32_t nPatterns) {
  checkSeed(seed);
  if (nPatterns > m_capacityPerSeed) {
    std::stringstream ss;
    ss << "CudaPatternRecordContainer: nPatterns " << nPatterns
       << " exceeds capacityPerSeed " << m_capacityPerSeed;
    throw std::out_of_range(ss.str());
  }

  m_host.nPatterns[seed] = nPatterns;
}

std::uint8_t CudaPatternRecordContainer::overflow(size_type seed) const {
  checkSeed(seed);

  return m_host.overflow[seed];
}

void CudaPatternRecordContainer::setOverflow(size_type seed,
                                             std::uint8_t overflow) {
  checkSeed(seed);

  m_host.overflow[seed] = overflow;
}

void CudaPatternRecordContainer::setPattern(size_type patternSlot,
                                            const CudaPatternRecordRow& row) {
  checkSlot(patternSlot);
  if (row.nHits > m_maxHitsPerPattern) {
    std::stringstream ss;
    ss << "CudaPatternRecordContainer: nHits " << row.nHits
       << " exceeds maxHitsPerPattern " << m_maxHitsPerPattern;
    throw std::out_of_range(ss.str());
  }
  const size_type p{patternSlot};

  m_host.sector[p] = row.sector;

  m_host.patPhi[p] = row.patPhi;
  m_host.patTheta[p] = row.patTheta;
  m_host.patPhiCov[p] = row.patPhiCov;

  m_host.meanNormResidual2[p] = row.meanNormResidual2;
  m_host.lastResidual[p] = row.lastResidual;
  m_host.lastResSigma[p] = row.lastResSigma;

  m_host.linePosX[p] = row.linePos.x();
  m_host.linePosY[p] = row.linePos.y();
  m_host.linePosZ[p] = row.linePos.z();

  m_host.lineDirX[p] = row.lineDir.x();
  m_host.lineDirY[p] = row.lineDir.y();
  m_host.lineDirZ[p] = row.lineDir.z();

  m_host.leverArm[p] = row.leverArm;

  m_host.bendPlaneNormX[p] = row.bendPlaneNorm.x();
  m_host.bendPlaneNormY[p] = row.bendPlaneNorm.y();
  m_host.bendPlaneNormZ[p] = row.bendPlaneNorm.z();

  m_host.nPrecisionLayers[p] = row.nPrecisionLayers;
  m_host.nTriggerLayers[p] = row.nTriggerLayers;
  m_host.nPhiLayers[p] = row.nPhiLayers;

  m_host.nHits[p] = row.nHits;

  m_host.lastInsertedHit[p] = row.lastInsertedHit;
  m_host.prevLayerHit[p] = row.prevLayerHit;
  m_host.lineAnchorHit[p] = row.lineAnchorHit;

  m_host.flags[p] = row.flags;
  m_host.status[p] = row.status;
}

CudaPatternRecordRow CudaPatternRecordContainer::pattern(
    size_type patternSlot) const {
  checkSlot(patternSlot);
  const size_type p{patternSlot};

  CudaPatternRecordRow row{};
  row.sector = m_host.sector[p];

  row.patPhi = m_host.patPhi[p];
  row.patTheta = m_host.patTheta[p];
  row.patPhiCov = m_host.patPhiCov[p];

  row.meanNormResidual2 = m_host.meanNormResidual2[p];
  row.lastResidual = m_host.lastResidual[p];
  row.lastResSigma = m_host.lastResSigma[p];

  row.linePos = {m_host.linePosX[p], m_host.linePosY[p], m_host.linePosZ[p]};
  row.lineDir = {m_host.lineDirX[p], m_host.lineDirY[p], m_host.lineDirZ[p]};
  row.leverArm = m_host.leverArm[p];
  row.bendPlaneNorm = {m_host.bendPlaneNormX[p], m_host.bendPlaneNormY[p],
                       m_host.bendPlaneNormZ[p]};

  row.nPrecisionLayers = m_host.nPrecisionLayers[p];
  row.nTriggerLayers = m_host.nTriggerLayers[p];
  row.nPhiLayers = m_host.nPhiLayers[p];

  row.nHits = m_host.nHits[p];

  row.lastInsertedHit = m_host.lastInsertedHit[p];
  row.prevLayerHit = m_host.prevLayerHit[p];
  row.lineAnchorHit = m_host.lineAnchorHit[p];

  row.flags = m_host.flags[p];
  row.status = m_host.status[p];

  return row;
}

void CudaPatternRecordContainer::setHit(size_type patternSlot, size_type local,
                                        std::uint32_t hitIndex,
                                        std::uint16_t globLayer) {
  const size_type slot{hitSlot(patternSlot, local)};

  m_host.hitIndex[slot] = hitIndex;
  m_host.globLayer[slot] = globLayer;
}

void CudaPatternRecordContainer::setNHits(size_type patternSlot,
                                          std::uint32_t nHits) {
  checkSlot(patternSlot);
  if (nHits > m_maxHitsPerPattern) {
    std::stringstream ss;
    ss << "CudaPatternRecordContainer: nHits " << nHits
       << " exceeds maxHitsPerPattern " << m_maxHitsPerPattern;
    throw std::out_of_range(ss.str());
  }

  m_host.nHits[patternSlot] = nHits;
}

std::uint32_t CudaPatternRecordContainer::nHits(size_type patternSlot) const {
  checkSlot(patternSlot);

  return m_host.nHits[patternSlot];
}

std::uint32_t CudaPatternRecordContainer::hitIndex(size_type patternSlot,
                                                   size_type local) const {
  return m_host.hitIndex[hitSlot(patternSlot, local)];
}

std::uint16_t CudaPatternRecordContainer::globLayer(size_type patternSlot,
                                                    size_type local) const {
  return m_host.globLayer[hitSlot(patternSlot, local)];
}

std::span<const std::uint32_t> CudaPatternRecordContainer::hitIndices(
    size_type patternSlot) const {
  const std::uint32_t used{nHits(patternSlot)};
  if (used == 0u) {
    return {};
  }

  return std::span<const std::uint32_t>{
      m_host.hitIndex.data() + hitSlot(patternSlot, 0), used};
}

void CudaPatternRecordContainer::moveToDevice(cudaStream_t stream) {
  clearDevice();

  allocateColumns(patternColumns, m_host, m_device);
  // A kernel reads these before it writes them; everything else of a slot is
  // valid only once the slot is counted in nPatterns
  zeroDeviceColumn(m_device.nPatterns, m_nSeeds, stream);
  zeroDeviceColumn(m_device.overflow, m_nSeeds, stream);
  zeroDeviceColumn(m_device.status, nSlots(), stream);

  m_device.nSeeds = static_cast<std::uint32_t>(m_nSeeds);
  m_device.capacityPerSeed = static_cast<std::uint32_t>(m_capacityPerSeed);
  m_device.maxHitsPerPattern = static_cast<std::uint32_t>(m_maxHitsPerPattern);
  m_onDevice = true;
}

void CudaPatternRecordContainer::moveToHost(cudaStream_t stream) {
  if (!m_onDevice) {
    return;
  }

  copyColumnsToHost(patternColumns, m_host, m_device, stream);
}

void CudaPatternRecordContainer::clearDevice() noexcept {
  freeColumns(patternColumns, m_host, m_device);

  m_device = {};
  m_onDevice = false;
}

CudaPatternRecordContainer::size_type CudaPatternRecordContainer::hitSlot(
    size_type patternSlot, size_type local) const {
  checkHit(patternSlot, local);

  return patternSlot * m_maxHitsPerPattern + local;
}

void CudaPatternRecordContainer::checkSeed(size_type seed) const {
  if (seed >= m_nSeeds) {
    std::stringstream ss;
    ss << "CudaPatternRecordContainer: seed " << seed << " is out of range for "
       << m_nSeeds << " seeds";
    throw std::out_of_range(ss.str());
  }
}

void CudaPatternRecordContainer::checkSlot(size_type patternSlot) const {
  if (patternSlot >= nSlots()) {
    std::stringstream ss;
    ss << "CudaPatternRecordContainer: pattern slot " << patternSlot
       << " is out of range for " << nSlots() << " slots";
    throw std::out_of_range(ss.str());
  }
}

void CudaPatternRecordContainer::checkHit(size_type patternSlot,
                                          size_type local) const {
  checkSlot(patternSlot);
  if (local >= m_maxHitsPerPattern) {
    std::stringstream ss;
    ss << "CudaPatternRecordContainer: hit slot " << local
       << " is out of range for maxHitsPerPattern " << m_maxHitsPerPattern;
    throw std::out_of_range(ss.str());
  }
}

}  // namespace ActsExamples
