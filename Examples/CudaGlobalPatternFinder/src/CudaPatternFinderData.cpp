// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/EventData/CudaPatternFinderData.hpp"

#include "ActsExamples/Utilities/CudaUtilities.hpp"

#include <algorithm>
#include <cstring>
#include <span>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

/// Every hit payload column; a new column is added here and in the two
/// structures only
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

  visit(host.geometryId, device.geometryId);

  visit(host.stationKey, device.stationKey);

  visit(host.station, device.station);
  visit(host.locLayer, device.locLayer);
  visit(host.flags, device.flags);
};

constexpr auto candidateInputColumns = [](auto& host, auto& device,
                                          auto&& visit) {
  visit(host.seedHitIndex, device.seedHitIndex);
  visit(host.seedSector, device.seedSector);
  visit(host.seedTheta, device.seedTheta);
  visit(host.candidateOffsets, device.candidateOffsets);
  visit(host.candidateIndices, device.candidateIndices);
};

constexpr auto candidateOutputColumns = [](auto& host, auto& device,
                                           auto&& visit) {
  visit(host.seedStatus, device.seedStatus);
  visit(host.seedPosition, device.seedPosition);
  visit(host.candidateGlobLayer, device.candidateGlobLayer);
  visit(host.candidateInfo, device.candidateInfo);
};

constexpr auto candidateColumns = [](auto& host, auto& device, auto&& visit) {
  candidateInputColumns(host, device, visit);
  candidateOutputColumns(host, device, visit);
};

constexpr auto patternSlotColumns = [](auto& host, auto& device, auto&& visit) {
  visit(host.sector, device.sector);

  visit(host.patPhi, device.patPhi);
  visit(host.patTheta, device.patTheta);
  visit(host.patPhiOffset, device.patPhiOffset);
  visit(host.bendPlaneCovPhiPhi, device.bendPlaneCovPhiPhi);
  visit(host.bendPlaneCovSS, device.bendPlaneCovSS);
  visit(host.bendPlaneCovPhiS, device.bendPlaneCovPhiS);
  visit(host.nDoF, device.nDoF);
  visit(host.lastNDoF, device.lastNDoF);
  visit(host.anchorRadius, device.anchorRadius);

  visit(host.sumNormResidual2, device.sumNormResidual2);
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

constexpr auto patternHitColumns = [](auto& host, auto& device, auto&& visit) {
  visit(host.hitIndex, device.hitIndex);
  visit(host.globLayer, device.globLayer);
};

constexpr auto patternSeedColumns = [](auto& host, auto& device, auto&& visit) {
  visit(host.nPatterns, device.nPatterns);
  visit(host.overflow, device.overflow);
};

constexpr auto patternColumns = [](auto& host, auto& device, auto&& visit) {
  patternSlotColumns(host, device, visit);
  patternHitColumns(host, device, visit);
  patternSeedColumns(host, device, visit);
};

template <typename T>
void zeroDeviceColumn(T* deviceColumn, std::size_t size, cudaStream_t stream) {
  if (size == 0 || deviceColumn == nullptr) {
    return;
  }
  ACTS_CUDA_CHECK(cudaMemsetAsync(deviceColumn, 0, size * sizeof(T), stream));
}

template <typename T>
std::size_t blockBytes(const T* /*column*/, std::size_t count) {
  return (count * sizeof(T) + 15u) & ~std::size_t{15u};
}

/// With a null block, only advances cursor
template <typename Columns, typename Host, typename Device, typename Count>
void carveColumns(Columns&& columns, Host& host, Device& device, char* block,
                  std::size_t& cursor, Count count) {
  columns(host, device, [&](auto& hostColumn, auto& deviceColumn) {
    const std::size_t n{count(hostColumn)};
    if (block != nullptr) {
      deviceColumn = n > 0u
                         ? reinterpret_cast<
                               std::remove_reference_t<decltype(deviceColumn)>>(
                               block + cursor)
                         : nullptr;
    }
    cursor += blockBytes(deviceColumn, n);
  });
}

/// All device columns in one stream-ordered allocation
template <typename Columns, typename Host, typename Device>
void* allocateColumns(Columns&& columns, Host& host, Device& device,
                      cudaStream_t stream) {
  auto hostSize = [](const auto& hostColumn) { return hostColumn.size(); };
  std::size_t bytes{0u};
  carveColumns(columns, host, device, nullptr, bytes, hostSize);
  if (bytes == 0u) {
    return nullptr;
  }
  char* block{nullptr};
  ACTS_CUDA_CHECK(
      cudaMallocAsync(reinterpret_cast<void**>(&block), bytes, stream));
  std::size_t cursor{0u};
  carveColumns(columns, host, device, block, cursor, hostSize);
  return block;
}

/// Only once all work on the block has completed
inline void freeBlock(void*& block) noexcept {
  if (block != nullptr) {
    // Called from noexcept destructors, so errors are ignored
    (void)cudaFreeAsync(block, cudaStreamPerThread);
    block = nullptr;
  }
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
      m_block{std::exchange(other.m_block, nullptr)},
      m_onDevice{std::exchange(other.m_onDevice, false)} {}

CudaHitPayloadContainer& CudaHitPayloadContainer::operator=(
    CudaHitPayloadContainer&& other) noexcept {
  if (this != &other) {
    clearDevice();

    m_size = std::exchange(other.m_size, 0);
    m_host = std::move(other.m_host);
    m_device = std::exchange(other.m_device, {});
    m_block = std::exchange(other.m_block, nullptr);
    m_onDevice = std::exchange(other.m_onDevice, false);
  }

  return *this;
}

CudaHitPayloadContainer::~CudaHitPayloadContainer() noexcept {
  clearDevice();
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

  m_host.geometryId[index] = row.geometryId;

  m_host.stationKey[index] = row.stationKey;

  m_host.station[index] = row.station;
  m_host.locLayer[index] = row.locLayer;
  m_host.flags[index] = row.flags;
}

void CudaHitPayloadContainer::moveToDevice(cudaStream_t stream) {
  clearDevice();

  m_block = allocateColumns(hitPayloadColumns, m_host, m_device, stream);
  copyColumnsToDevice(hitPayloadColumns, m_host, m_device, stream);

  m_onDevice = true;
}

void CudaHitPayloadContainer::clearDevice() noexcept {
  freeBlock(m_block);

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
      m_block{std::exchange(other.m_block, nullptr)},
      m_onDevice{std::exchange(other.m_onDevice, false)} {
  other.m_host = {};
}

CudaCandidateListContainer& CudaCandidateListContainer::operator=(
    CudaCandidateListContainer&& other) noexcept {
  if (this != &other) {
    clearDevice();

    m_host = std::move(other.m_host);
    m_device = std::exchange(other.m_device, {});
    m_block = std::exchange(other.m_block, nullptr);
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
  m_host.candidateInfo.resize(m_host.candidateIndices.size(), 0u);
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

  m_block = allocateColumns(candidateColumns, m_host, m_device, stream);
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
  freeBlock(m_block);

  m_device = {};
  m_onDevice = false;
  m_deviceOnly = false;
}

void CudaCandidateListContainer::allocateDevice(size_type nSeeds,
                                                size_type nCandidates,
                                                cudaStream_t stream) {
  clearDevice();
  m_host = {};
  auto count = [&](const auto& hostColumn) -> std::size_t {
    const void* column{&hostColumn};
    if (column == &m_host.candidateOffsets) {
      return nSeeds + 1u;
    }
    if (column == &m_host.candidateIndices ||
        column == &m_host.candidateGlobLayer ||
        column == &m_host.candidateInfo) {
      return nCandidates;
    }
    return nSeeds;
  };
  std::size_t bytes{0u};
  carveColumns(candidateColumns, m_host, m_device, nullptr, bytes, count);
  char* block{nullptr};
  ACTS_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&block),
                                  std::max<std::size_t>(bytes, 1u), stream));
  std::size_t cursor{0u};
  carveColumns(candidateColumns, m_host, m_device, block, cursor, count);
  m_block = block;
  // Zeroed, so that a seed the kernel has not reached reads as pending
  candidateOutputColumns(
      m_host, m_device, [&](const auto& hostColumn, auto& deviceColumn) {
        zeroDeviceColumn(deviceColumn, count(hostColumn), stream);
      });
  m_device.nSeeds = static_cast<std::uint32_t>(nSeeds);
  m_device.nCandidates = static_cast<std::uint32_t>(nCandidates);
  m_onDevice = true;
  m_deviceOnly = true;
}

void CudaCandidateListContainer::checkSeed(size_type seed) const {
  if (seed >= nSeeds()) {
    std::stringstream ss;
    ss << "CudaCandidateListContainer: seed " << seed << " is out of range for "
       << nSeeds() << " seeds";
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
      m_block{std::exchange(other.m_block, nullptr)},
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
    m_block = std::exchange(other.m_block, nullptr);
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

CudaPatternRecordRow CudaPatternRecordContainer::pattern(
    size_type patternSlot) const {
  checkSlot(patternSlot);
  const size_type p{patternSlot};

  CudaPatternRecordRow row{};
  row.sector = m_host.sector[p];

  row.patPhi = m_host.patPhi[p];
  row.patTheta = m_host.patTheta[p];
  row.patPhiOffset = m_host.patPhiOffset[p];
  row.bendPlaneCovPhiPhi = m_host.bendPlaneCovPhiPhi[p];
  row.bendPlaneCovSS = m_host.bendPlaneCovSS[p];
  row.bendPlaneCovPhiS = m_host.bendPlaneCovPhiS[p];
  row.nDoF = m_host.nDoF[p];
  row.lastNDoF = m_host.lastNDoF[p];
  row.anchorRadius = m_host.anchorRadius[p];

  row.sumNormResidual2 = m_host.sumNormResidual2[p];
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

void CudaPatternRecordContainer::clearDevice() noexcept {
  freeBlock(m_block);

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

CudaPinnedBuffer::~CudaPinnedBuffer() {
  if (m_data != nullptr) {
    (void)cudaFreeHost(m_data);
  }
}

char* CudaPinnedBuffer::reserve(std::size_t bytes) {
  if (bytes > m_size) {
    if (m_data != nullptr) {
      ACTS_CUDA_CHECK(cudaFreeHost(m_data));
      m_data = nullptr;
    }
    const std::size_t size{std::max(bytes, 2u * m_size)};
    m_size = 0u;
    ACTS_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&m_data), size));
    m_size = size;
  }
  return m_data;
}

CudaDeviceBuffer::~CudaDeviceBuffer() {
  if (m_data != nullptr) {
    (void)cudaFree(m_data);
  }
}

char* CudaDeviceBuffer::reserve(std::size_t bytes) {
  if (bytes > m_size) {
    if (m_data != nullptr) {
      ACTS_CUDA_CHECK(cudaFree(m_data));
      m_data = nullptr;
    }
    const std::size_t size{std::max(bytes, 2u * m_size)};
    m_size = 0u;
    ACTS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_data), size));
    m_size = size;
  }
  return m_data;
}

void CudaHitPayloadContainer::placeOnDevice(size_type size,
                                            CudaStagedBlock& block) {
  const bool place{block.host != nullptr};
  if (place) {
    clearDevice();
    m_size = size;
    m_host = {};
  }
  hitPayloadColumns(m_host, m_device,
                    [&](auto& /*hostColumn*/, auto& deviceColumn) {
                      using T = std::remove_pointer_t<
                          std::remove_reference_t<decltype(deviceColumn)>>;
                      const auto columns = block.take<T>(size);
                      if (place) {
                        deviceColumn = columns.second;
                      }
                    });
  m_onDevice = m_onDevice || place;
}

void CudaPatternRecordContainer::copyFromHostCopy(
    const CudaPatternRecordArrays& source, const char* deviceBase,
    const char* hostCopy) {
  if (source.maxHitsPerPattern != m_maxHitsPerPattern) {
    throw std::invalid_argument(
        "CudaPatternRecordContainer::copyFromHostCopy: the hit slots per "
        "pattern differ");
  }
  CudaPatternRecordArrays& sourceColumns{
      const_cast<CudaPatternRecordArrays&>(source)};
  patternColumns(
      m_host, sourceColumns, [&](auto& hostColumn, const auto& deviceColumn) {
        if (hostColumn.empty()) {
          return;
        }
        const char* at{hostCopy + (reinterpret_cast<const char*>(deviceColumn) -
                                   deviceBase)};
        std::memcpy(hostColumn.data(), at,
                    hostColumn.size() * sizeof(hostColumn[0]));
      });
}

}  // namespace ActsExamples
