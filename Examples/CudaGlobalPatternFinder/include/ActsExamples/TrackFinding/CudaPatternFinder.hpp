// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "ActsExamples/EventData/CudaPatternFinderData.hpp"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

/// The arithmetic of the global pattern finder, written once for the host and
/// the device. Every function is a copy of the CPU code it names, with the
/// operations in the same order, so that a kernel reproduces the CPU numbers
/// bit for bit as long as floating-point contraction is off on both sides. A
/// change to the CPU formulas has to be made here as well; the unit tests
/// compare the two.
namespace ActsExamples::CudaPatternMath {

/// @brief Acts::s_epsilon
constexpr double epsilon = 3. * DBL_EPSILON;

constexpr double pi = 3.14159265358979323846;

/// @brief Plain three-vector, usable in a kernel
struct Vector3 {
  double x{0.};
  double y{0.};
  double z{0.};
};

__host__ __device__ inline Vector3 operator+(const Vector3& a,
                                             const Vector3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
__host__ __device__ inline Vector3 operator-(const Vector3& a,
                                             const Vector3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__host__ __device__ inline Vector3 operator*(double s, const Vector3& a) {
  return {s * a.x, s * a.y, s * a.z};
}
__host__ __device__ inline Vector3 operator/(const Vector3& a, double s) {
  return {a.x / s, a.y / s, a.z / s};
}
__host__ __device__ inline double dot(const Vector3& a, const Vector3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline double norm(const Vector3& a) {
  return std::sqrt(dot(a, a));
}
__host__ __device__ inline double square(double x) {
  return x * x;
}
/// @brief Acts::VectorHelpers::perp and Acts::fastHypot(x, y)
__host__ __device__ inline double perp(const Vector3& a) {
  return std::sqrt(a.x * a.x + a.y * a.y);
}

/// @brief Acts::VectorHelpers::phi
__host__ __device__ inline double phi(const Vector3& a) {
  return std::atan2(a.y, a.x);
}

/// @brief Acts::VectorHelpers::theta
__host__ __device__ inline double theta(const Vector3& a) {
  return std::atan2(perp(a), a.z);
}

/// @brief Acts::makeDirectionFromPhiTheta
__host__ __device__ inline Vector3 directionFromPhiTheta(double phi,
                                                         double theta) {
  const double cosTheta{std::cos(theta)};
  const double sinTheta{std::sin(theta)};
  return Vector3{std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta};
}

/// @brief Acts::detail::difference_periodic
__host__ __device__ inline double differencePeriodic(double lhs, double rhs,
                                                     double range) {
  double delta{std::fmod(lhs - rhs, range)};
  if ((2. * delta) < -range) {
    delta += range;
  } else if (range <= (2. * delta)) {
    delta -= range;
  }
  return delta;
}

/// @brief The quantities of one hit that the arithmetic reads
struct Hit {
  Vector3 position{};
  Vector3 sensorDirection{};
  Vector3 etaMeasurementDirection{};
  Vector3 phiMeasurementDirection{};
  double localPositionY{0.};
  double etaCovariance{0.};
  double phiCovariance{0.};
  double driftCovariance{0.};
  double phiVariance{0.};
  std::uint32_t stationKey{0u};
  std::uint8_t station{0u};
  std::uint8_t locLayer{0u};
  std::uint8_t flags{0u};
};

/// @brief Reads one row of the hit payload columns
__host__ __device__ inline Hit loadHit(const CudaHitPayloadArrays& hits,
                                       std::uint32_t row) {
  Hit hit{};
  hit.position = {hits.positionX[row], hits.positionY[row],
                  hits.positionZ[row]};
  hit.sensorDirection = {hits.sensorDirectionX[row], hits.sensorDirectionY[row],
                         hits.sensorDirectionZ[row]};
  hit.etaMeasurementDirection = {hits.etaMeasurementDirectionX[row],
                                 hits.etaMeasurementDirectionY[row],
                                 hits.etaMeasurementDirectionZ[row]};
  hit.phiMeasurementDirection = {hits.phiMeasurementDirectionX[row],
                                 hits.phiMeasurementDirectionY[row],
                                 hits.phiMeasurementDirectionZ[row]};
  hit.localPositionY = hits.localPositionY[row];
  hit.etaCovariance = hits.etaCovariance[row];
  hit.phiCovariance = hits.phiCovariance[row];
  hit.driftCovariance = hits.driftCovariance[row];
  hit.phiVariance = hits.phiVariance[row];
  hit.stationKey = hits.stationKey[row];
  hit.station = hits.station[row];
  hit.locLayer = hits.locLayer[row];
  hit.flags = hits.flags[row];
  return hit;
}

/// @brief The same, from a row of the host container
inline Hit makeHit(const CudaHitPayloadRow& row) {
  auto toVector = [](const Acts::Vector3& v) {
    return Vector3{v.x(), v.y(), v.z()};
  };
  Hit hit{};
  hit.position = toVector(row.position);
  hit.sensorDirection = toVector(row.sensorDirection);
  hit.etaMeasurementDirection = toVector(row.etaMeasurementDirection);
  hit.phiMeasurementDirection = toVector(row.phiMeasurementDirection);
  hit.localPositionY = row.localPositionY;
  hit.etaCovariance = row.etaCovariance;
  hit.phiCovariance = row.phiCovariance;
  hit.driftCovariance = row.driftCovariance;
  hit.phiVariance = row.phiVariance;
  hit.stationKey = row.stationKey;
  hit.station = row.station;
  hit.locLayer = row.locLayer;
  hit.flags = row.flags;
  return hit;
}

/// @brief HitPayload::intrinsicVariance: the hit position covariance
///        contracted along a direction. Not defined for a phi-only hit, where
///        the CPU throws; it returns zero here, and a phi-only hit never
///        reaches the pattern building.
__host__ __device__ inline double intrinsicVariance(const Hit& hit,
                                                    const Vector3& c) {
  const bool measuresPhi{(hit.flags & CudaHitFlags::measuresPhi) != 0u};
  if ((hit.flags & CudaHitFlags::straw) != 0u) {
    const double dotProdSq{square(dot(hit.sensorDirection, c))};
    const double etaTerm{hit.driftCovariance * (dot(c, c) - dotProdSq)};
    if (measuresPhi) {
      return etaTerm + dotProdSq * hit.phiCovariance;
    }
    return etaTerm;
  }
  if ((hit.flags & CudaHitFlags::measuresEta) == 0u) {
    return 0.;
  }
  const double etaTerm{hit.etaCovariance *
                       square(dot(hit.etaMeasurementDirection, c))};
  if (measuresPhi) {
    return etaTerm +
           hit.phiCovariance * square(dot(hit.phiMeasurementDirection, c));
  }
  return etaTerm;
}

/// @brief PatternState::projToPhiPlane: the hit moved along its sensor onto
///        the bending plane through the origin, Acts::PlanarHelper::
///        intersectPlane. A sensor parallel to the plane has no intersection;
///        the CPU then takes the position of an invalid intersection, which
///        is the origin, and so does this
__host__ __device__ inline Vector3 projectToPhiPlane(
    const Hit& hit, const Vector3& bendPlaneNorm) {
  const double offset{dot(Vector3{}, bendPlaneNorm)};
  const double normDot{dot(bendPlaneNorm, hit.sensorDirection)};
  if (std::abs(normDot) < DBL_EPSILON) {
    return Vector3{};
  }
  const double path{(offset - dot(hit.position, bendPlaneNorm)) / normDot};
  return hit.position + path * hit.sensorDirection;
}

/// @brief The pattern line, as PatternState holds it
struct Line {
  Vector3 position{};
  Vector3 direction{};
  double leverArm{0.};
  bool useBeamspot{false};
};

/// @brief The beam spot, PatternState::BeamspotInfo
struct Beamspot {
  Vector3 position{};
  double radius{0.};
  double length{0.};
};

/// @brief PatternState::updateLineParameters: the line through the projected
///        anchor and last hit, or through the beam spot when the two are
///        closer than minHitDistance
/// @param anchor The anchor hit projected onto the bending plane
/// @param last The last hit projected onto the bending plane
__host__ __device__ inline Line lineThroughHits(const Vector3& anchor,
                                                const Vector3& last,
                                                double minHitDistance,
                                                const Beamspot& beamspot) {
  Line line{};
  Vector3 d{last - anchor};
  line.leverArm = norm(d);
  line.useBeamspot = line.leverArm < minHitDistance;
  if (line.useBeamspot) {
    line.position = beamspot.position;
    d = last - beamspot.position;
    line.leverArm = norm(d);
  } else {
    line.position = anchor;
  }
  line.direction = d / line.leverArm;
  return line;
}

/// @brief Residual of a test hit against the pattern line and its uncertainty
struct LineResidual {
  double residual{0.};
  double sigma{0.};
};

/// @brief PatternState::computeLineResidual
/// @param line The pattern line
/// @param bendPlaneNorm The normal of the bending plane
/// @param patPhiCov The variance of the pattern phi
/// @param nPhiLayers The number of phi layers of the pattern
/// @param anchor The line anchor hit
/// @param last The last inserted hit
/// @param test The hit tested against the line
/// @param beamspot The beam spot
__host__ __device__ inline LineResidual lineResidual(
    const Line& line, const Vector3& bendPlaneNorm, double patPhiCov,
    std::uint32_t nPhiLayers, const Hit& anchor, const Hit& last,
    const Hit& test, const Beamspot& beamspot) {
  LineResidual res{};

  const bool projectTestHit{(test.flags & CudaHitFlags::measuresPhi) == 0u ||
                            nPhiLayers == 0u};
  const Vector3 testPos{projectTestHit ? projectToPhiPlane(test, bendPlaneNorm)
                                       : test.position};
  const Vector3 K{testPos - line.position};
  const double KdotD{dot(K, line.direction)};

  const Vector3 R{K - KdotD * line.direction};
  res.residual = norm(R);
  if (res.residual < epsilon) {
    res.residual = DBL_MAX;
    res.sigma = 0.;
    return res;
  }
  const Vector3 resDir{R / res.residual};
  const double alpha{KdotD / line.leverArm};

  double phiPlaneDerivativeAcc{0.};
  double residualCovAcc{0.};

  auto covarianceTerm = [&](const Hit& hit, const Vector3& pos,
                            const double preFactor, bool isProjected) {
    if (!isProjected) {
      residualCovAcc += square(preFactor) * intrinsicVariance(hit, resDir);
      return;
    }
    const Vector3& sensorDir{hit.sensorDirection};
    const double projFactor{dot(sensorDir, resDir) /
                            dot(sensorDir, bendPlaneNorm)};
    const Vector3 trfDir{resDir - projFactor * bendPlaneNorm};

    residualCovAcc += square(preFactor) * intrinsicVariance(hit, trfDir);
    phiPlaneDerivativeAcc += preFactor * perp(pos) * projFactor;
  };

  if (line.useBeamspot) {
    const double covS1 = beamspot.length * square(resDir.z) +
                         beamspot.radius * (1 - square(resDir.z));
    residualCovAcc += square(alpha - 1) * covS1;
  } else {
    covarianceTerm(anchor, line.position, alpha - 1, true);
  }

  const Vector3 pos2{line.position + line.leverArm * line.direction};
  covarianceTerm(last, pos2, -alpha, true);

  covarianceTerm(test, testPos, 1., projectTestHit);

  res.sigma =
      std::sqrt(residualCovAcc + square(phiPlaneDerivativeAcc) * patPhiCov);
  return res;
}

/// @brief Station and layer numbers, as Muon::MuonStationIndex numbers them
namespace Station {
constexpr std::uint8_t BI = 0u, BM = 1u, BO = 2u, BE = 3u, EI = 4u, EM = 5u,
                       EO = 6u, EE = 7u;
/// @brief PatternTopology::nGroups
constexpr std::uint8_t nGroups = 8u;
}  // namespace Station
namespace Layer {
constexpr std::uint8_t Inner = 0u, Middle = 1u, Outer = 2u, Extended = 3u,
                       BarrelExtended = 4u;
}

/// @brief ActsExamples::isBarrel
__host__ __device__ inline bool isBarrel(std::uint8_t station) {
  return station <= Station::BE;
}

/// @brief ActsExamples::toLayerIndex
__host__ __device__ inline std::uint8_t layerOf(std::uint8_t station) {
  switch (station) {
    case Station::BI:
    case Station::EI:
      return Layer::Inner;
    case Station::BM:
    case Station::EM:
      return Layer::Middle;
    case Station::BO:
    case Station::EO:
      return Layer::Outer;
    case Station::BE:
      return Layer::BarrelExtended;
    default:
      return Layer::Extended;
  }
}

/// @brief PatternTopology::groupIndex
__host__ __device__ inline std::uint8_t groupIndex(const Hit& hit) {
  return hit.station;
}

/// @brief PatternTopology::layerSorter: whether hit 1 comes before hit 2. Two
///        hits in BO and EO, which the CPU rejects with an exception, are
///        ordered as false here
__host__ __device__ inline bool layerSorter(const Hit& hit1, const Hit& hit2) {
  const std::uint8_t st1{hit1.station};
  const std::uint8_t st2{hit2.station};
  if (st1 == st2) {
    if (hit1.stationKey == hit2.stationKey) {
      if (hit1.locLayer == hit2.locLayer) {
        return hit1.localPositionY < hit2.localPositionY;
      }
      return hit1.locLayer < hit2.locLayer;
    }
    const double delta{isBarrel(st1) ? perp(hit1.position) - perp(hit2.position)
                                     : std::abs(hit1.position.z) -
                                           std::abs(hit2.position.z)};
    if (std::abs(delta) <= epsilon) {
      return hit1.localPositionY < hit2.localPositionY;
    }
    return delta < 0.;
  }
  const std::uint8_t layer1{layerOf(st1)};
  const std::uint8_t layer2{layerOf(st2)};
  if (layer1 == layer2) {
    if (layer1 == Layer::Middle) {
      return st1 == Station::BM;
    }
    if (layer1 == Layer::Inner) {
      return perp(hit1.position) < perp(hit2.position);
    }
    return false;
  }
  if (layer1 == Layer::Inner || layer2 == Layer::Inner) {
    return layer1 == Layer::Inner;
  }
  if (layer1 == Layer::Outer || layer2 == Layer::Outer) {
    return layer2 == Layer::Outer;
  }
  if (layer1 == Layer::BarrelExtended || layer2 == Layer::BarrelExtended) {
    return layer1 == Layer::BarrelExtended;
  }
  if (layer1 == Layer::Extended) {
    return st2 == Station::EM;
  }
  return st1 == Station::BM;
}

/// @brief PatternTopology::sameLayer
__host__ __device__ inline bool sameLayer(const Hit& hit1, const Hit& hit2) {
  if (hit1.station != hit2.station) {
    return false;
  }
  if (hit1.locLayer == hit2.locLayer && hit1.stationKey == hit2.stationKey) {
    return true;
  }
  const double delta{isBarrel(hit1.station)
                         ? perp(hit1.position) - perp(hit2.position)
                         : std::abs(hit1.position.z) -
                               std::abs(hit2.position.z)};
  return std::abs(delta) <= epsilon;
}

/// @brief The expanded sectors of ActsExamples::ExpandedSector: the 16 muon
///        sectors and the 16 overlaps between neighbours, numbered 0 to 31.
///        Even numbers are sector centres, odd ones the overlap with the next
///        sector
namespace ExpandedSector {
constexpr int nMsSectors = 16;
constexpr int nExpanded = 2 * nMsSectors;
constexpr double oneEighthOfPi = pi / 8.;
constexpr double overlap = 0.1 * oneEighthOfPi;

/// @brief sectorMap::sectorSize: half size of a muon sector, without overlap
__host__ __device__ inline double msSize(int msSector) {
  return msSector % 2 == 0 ? 0.4 * oneEighthOfPi : 0.6 * oneEighthOfPi;
}

/// @brief sectorMap::sectorWidth
__host__ __device__ inline double msWidth(int msSector) {
  return msSize(msSector) + overlap;
}

/// @brief sectorMap::sectorPhi
__host__ __device__ inline double msPhi(int msSector) {
  if (msSector < 10) {
    return pi * (msSector - 1) / 8.;
  }
  return -pi * (2 - (msSector - 1) / 8.);
}

/// @brief sectorMap::insideSector
__host__ __device__ inline bool insideMsSector(int msSector, double phi) {
  double dphi{phi - msPhi(msSector)};
  if (dphi > pi) {
    dphi -= 2 * pi;
  }
  if (dphi < -pi) {
    dphi += 2 * pi;
  }
  return !(dphi < -msWidth(msSector)) && !(dphi > msWidth(msSector));
}

/// @brief ExpandedSector::msSector
__host__ __device__ inline int msSector(int sector) {
  return sector < 2 ? nMsSectors : sector / 2;
}

/// @brief ExpandedSector::adjacentMsSector
__host__ __device__ inline int adjacentMsSector(int sector) {
  const int ms{msSector(sector)};
  if (sector % 2 == 0) {
    return ms;
  }
  return ms == nMsSectors ? 1 : ms + 1;
}

/// @brief ExpandedSector::phi
__host__ __device__ inline double phi(int sector) {
  const int s1{msSector(sector)};
  const int s2{adjacentMsSector(sector)};
  if (s1 == s2) {
    return msPhi(s1);
  }
  // The overlap starts at the upper edge of the first sector of the pair
  double phiO{msPhi(s1) + msSize(s1)};
  if (phiO > pi) {
    phiO -= 2 * pi;
  }
  return phiO;
}

/// @brief ExpandedSector::sectorSize
__host__ __device__ inline double size(int sector) {
  const int s1{msSector(sector)};
  if (s1 == adjacentMsSector(sector)) {
    return msSize(s1);
  }
  return msWidth(s1) - msSize(s1);
}

/// @brief ExpandedSector::insideSector
__host__ __device__ inline bool inside(int sector, double phi) {
  return insideMsSector(msSector(sector), phi) &&
         insideMsSector(adjacentMsSector(sector), phi);
}

/// @brief ExpandedSector::isNeighbour
__host__ __device__ inline bool isNeighbour(int sector1, int sector2) {
  const int dS{std::abs(sector2 - sector1)};
  return (dS < nExpanded - dS ? dS : nExpanded - dS) <= 1;
}
}  // namespace ExpandedSector

}  // namespace ActsExamples::CudaPatternMath

namespace ActsExamples {

/// @brief The constants of the GlobalPatternFinder config that the kernels
///        read
struct CudaPatternFinderConfig {
  double nResidualSigma{3.};
  double nPhiSigma{3.};
  double minHitDistance4Line{40.};
  double maxThetaSizeOverlap{0.05};
  double lowConfidenceResSigma{50.};
  double meanNormRes2Cut{0.2};
  std::uint32_t minStripEtaLayers{3u};
  std::uint32_t minPrecisionLayers{0u};
  std::uint32_t minGroups{2u};
  std::uint32_t minGroupLayers{4u};
  std::uint32_t maxMissLayersInGroup{2u};
  std::uint32_t maxSeedAttempts{2u};
  CudaPatternMath::Beamspot beamspot{};
};

/// @brief Sizes of the device buffers of the pattern finding. They bound the
///        memory, not the result: what does not fit is built again larger
struct CudaPatternFinderSizes {
  /// @brief Seeds built at once before their patterns are accepted
  std::size_t batchSize{1024u};
  /// @brief Patterns a seed holds at once in the first build
  std::size_t capacityPerSeed{8u};
  /// @brief The same, for the seeds that overflowed the first build
  std::size_t retryCapacityPerSeed{64u};
  /// @brief Hits of one pattern
  std::size_t maxHitsPerPattern{64u};
  /// @brief Accepted patterns of the event
  std::size_t maxPatterns{4096u};
  /// @brief Runs of the event, each with doubled retry sizes, before giving
  ///        up
  std::size_t maxAttempts{4u};
};

/// @brief Where the time of one event went, in microseconds, and what the
///        event held. Host stages are wall time, device stages are measured
///        with CUDA events on the stream.
struct CudaPatternFinderTiming {
  /// @brief Host: the search tree
  double treeUs{0.};
  /// @brief Host: the hit columns and the candidate lists
  double fillUs{0.};
  /// @brief Device: copy of the hits and the lists to the device
  double uploadUs{0.};
  /// @brief Device: seed preparation, builds and accepts, all attempts
  double kernelsUs{0.};
  /// @brief Device: copy of the event's patterns to the host
  double downloadUs{0.};
  /// @brief Host: phi-only hits and the conversion of the patterns
  double tailUs{0.};
  std::size_t nSeeds{0u};
  std::size_t nCandidates{0u};
  /// @brief Patterns in eta accepted on the device
  std::size_t nPatternsInEta{0u};
  /// @brief Runs of the event on the device; more than one if it overflowed
  std::size_t nAttempts{0u};
};

/// @brief Copies the constants out of the CPU config
/// @tparam Config ActsExamples::GlobalPatternFinder_t::Config
/// @tparam Beamspot ActsExamples::GlobalPatternFinder_t::BeamspotInfo
template <typename Config, typename Beamspot>
CudaPatternFinderConfig makeCudaPatternFinderConfig(const Config& cfg,
                                                    const Beamspot& beamspot) {
  return CudaPatternFinderConfig{
      .nResidualSigma = cfg.nResidualSigma,
      .nPhiSigma = cfg.nPhiSigma,
      .minHitDistance4Line = cfg.minHitDistance4Line,
      .maxThetaSizeOverlap = cfg.maxThetaSizeOverlap,
      .lowConfidenceResSigma = cfg.lowConfidenceResSigma,
      .meanNormRes2Cut = cfg.meanNormRes2Cut,
      .minStripEtaLayers = cfg.minStripEtaLayers,
      .minPrecisionLayers = cfg.minPrecisionLayers,
      .minGroups = cfg.minGroups,
      .minGroupLayers = cfg.minGroupLayers,
      .maxMissLayersInGroup = cfg.maxMissLayersInGroup,
      .maxSeedAttempts = cfg.maxSeedAttempts,
      .beamspot = {
          {beamspot.position.x(), beamspot.position.y(), beamspot.position.z()},
          beamspot.radius,
          beamspot.length}};
}

/// @brief Runs the seed preparation of the CPU seed loop on the device, one
///        thread per seed: the candidate count, the group count, the sort by
///        layer and the global layer numbering, with their cuts.
///
/// Each candidate list is sorted in place with a stable insertion sort, so
/// ties of PatternTopology::layerSorter keep the order of the range search.
/// The status, the global layers and the position of the seed hit in its
/// sorted list are written to the output columns of the lists; the position is
/// cudaInvalidHitPosition if the seed hit is not in its own list. Both
/// containers must be on the device. The call waits only for the stream.
void prepareCudaSeeds(const CudaHitPayloadContainer& hits,
                      CudaCandidateListContainer& lists,
                      const CudaPatternFinderConfig& cfg, cudaStream_t stream);

/// @brief Builds the patterns of a range of seeds on the device, one thread
///        per seed, with the logic of GlobalPatternFinder::findPatternsInEta
///        after the seed checks: the forward and the backward extension, the
///        overlap removal among the patterns of the seed and the pattern cuts.
///
/// The lists must have been through prepareCudaSeeds. Entry b of the records
/// is seed firstSeed + b, and gets its patterns, already finalized. A seed
/// that needs more patterns at once than the records have slots per seed, or
/// more hits in a pattern than they have hit slots, sets its overflow flag and
/// writes no pattern. With rebuildOverflowOf, only the seeds that overflowed
/// those records are built. The call waits only for the stream.
void buildCudaPatterns(
    const CudaHitPayloadContainer& hits,
    const CudaCandidateListContainer& lists, const CudaPatternFinderConfig& cfg,
    std::size_t firstSeed, CudaPatternRecordContainer& patterns,
    cudaStream_t stream,
    const CudaPatternRecordContainer* rebuildOverflowOf = nullptr);

/// @brief Finds the patterns in eta of an event on the device.
///
/// The seeds are prepared, then built in batches in tree order; seeds that
/// overflow are built again with more room. After each batch, one thread
/// accepts its patterns seed by seed: a seed whose hit is already in
/// maxSeedAttempts accepted patterns is skipped, and each new pattern either
/// replaces the accepted patterns it overlaps or loses to one of them. If the
/// event still overflows, it is run again with doubled sizes. The hits and the
/// lists must be on the device. Returns the event's patterns on the host as
/// the slots of entry 0: the accepted ones have status ok, the dropped ones
/// status rejected. The call synchronizes the stream.
CudaPatternRecordContainer findCudaPatternsInEta(
    const CudaHitPayloadContainer& hits, CudaCandidateListContainer& lists,
    const CudaPatternFinderConfig& cfg, CudaPatternFinderSizes sizes,
    cudaStream_t stream, CudaPatternFinderTiming* timing = nullptr);

}  // namespace ActsExamples

// The driver works with the CPU types of the pattern finder, which the device
// code does not need
#ifndef __CUDACC__
#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "ActsExamples/TrackFinding/GlobalPatternFinderAlgorithm.hpp"

#include <memory>
#include <string>
#include <vector>

namespace ActsExamples {

/// @brief GlobalPatternFinder_t::findPatterns on the GPU. The patterns in eta
///        are found by findCudaPatternsInEta; the phi-only hits are added and
///        the patterns converted on the host, as the CPU does it.
class CudaGlobalPatternFinder {
 public:
  using OutputPattern = GlobalPatternFinder_t::OutputPattern;
  using PatternState = GlobalPatternFinder_t::PatternState;
  using BeamspotInfo = GlobalPatternFinder_t::BeamspotInfo;

  struct Config {
    /// @brief The configuration of the CPU pattern finder
    GlobalPatternFinder_t::Config finder{};
    /// @brief The sizes of the device buffers
    CudaPatternFinderSizes sizes{};
  };

  explicit CudaGlobalPatternFinder(
      Config cfg,
      std::unique_ptr<const Acts::Logger> logger = Acts::getDefaultLogger(
          "CudaGlobalPatternFinder", Acts::Logging::INFO));

  /// @brief Finds the patterns of an event
  /// @param gctx Geometry context
  /// @param treeData The hit payloads and the search tree over them
  /// @param seedSelector Selector of the seed hits
  /// @param onlyPhiProvider Provider of the phi-only hits of a pattern
  /// @param beamspot The beam spot
  /// @param timing Filled with the time of each stage, if given
  /// @return The patterns, in the order they were accepted
  std::vector<OutputPattern> findPatterns(
      const Acts::GeometryContext& gctx, const SearchTreeData& treeData,
      const SeedSelector& seedSelector,
      const OnlyPhiHitsProvider& onlyPhiProvider, const BeamspotInfo& beamspot,
      CudaPatternFinderTiming* timing = nullptr) const;

 private:
  /// @brief A pattern found on the device as the CPU's PatternState
  PatternState makePatternState(const Acts::GeometryContext& gctx,
                                const std::vector<HitPayload>& payloads,
                                const CudaPatternRecordContainer& event,
                                std::size_t slot) const;

  /// @brief GlobalPatternFinder::addPhiOnlyHits
  void addPhiOnlyHits(const Acts::GeometryContext& gctx,
                      const OnlyPhiHitsProvider& onlyPhiProvider,
                      std::vector<PatternState>& patterns) const;

  const Acts::Logger& logger() const { return *m_logger; }

  Config m_cfg;
  std::unique_ptr<const Acts::Logger> m_logger;
};

/// @brief GlobalPatternFinderAlgorithm with the pattern finding on the GPU.
///        It reads the same space points and writes the same patterns; the
///        search tree and the conversion of the patterns are those of the CPU
///        algorithm.
class CudaGlobalPatternFinderAlgorithm final : public IAlgorithm {
 public:
  struct Config : GlobalPatternFinderAlgorithm::Config {
    /// @brief CudaPatternFinderSizes
    std::size_t batchSize{1024u};
    std::size_t capacityPerSeed{8u};
    std::size_t retryCapacityPerSeed{64u};
    std::size_t maxHitsPerPattern{64u};
    std::size_t maxPatterns{4096u};
    std::size_t maxAttempts{4u};
    /// @brief Key of the CudaPatternFinderTiming of each event in the event
    ///        store; nothing is written if empty
    std::string outTiming{};
  };

  CudaGlobalPatternFinderAlgorithm(const Config& cfg,
                                   std::unique_ptr<const Acts::Logger> logger);

  ProcessCode execute(const AlgorithmContext& ctx) const override;

  const Config& config() const { return m_cfg; }

 private:
  /// @brief GlobalPatternFinderAlgorithm::constructTree
  SearchTreeData constructTree(
      const Acts::GeometryContext& gctx,
      const MuonSpacePointContainer& spacePoints) const;

  Config m_cfg;
  std::unique_ptr<SeedSelector> m_seedSelector{};
  std::unique_ptr<CudaGlobalPatternFinder> m_finder{};
  OnlyPhiHitsProvider m_onlyPhiProvider{};

  ReadDataHandle<MuonSpacePointContainer> m_inSpacePoints{this,
                                                          "InSpacePoints"};
  WriteDataHandle<MuonGlobalPatternContainer> m_outPatterns{this,
                                                            "OutPatterns"};
  WriteDataHandle<CudaPatternFinderTiming> m_outTiming{this, "OutTiming"};
};

}  // namespace ActsExamples
#endif
