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
}
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

}  // namespace ActsExamples::CudaPatternMath
