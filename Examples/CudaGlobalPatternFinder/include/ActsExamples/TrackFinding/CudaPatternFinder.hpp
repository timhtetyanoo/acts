// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "ActsExamples/EventData/CudaMuonSpacePoint.hpp"
#include "ActsExamples/EventData/CudaPatternFinderData.hpp"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <type_traits>

#include <cuda_runtime.h>

/// The arithmetic of the global pattern finder, written once for the host and
/// the device. Every function is a copy of the CPU code it names, with the
/// operations in the same order, so that in double precision it reproduces the
/// CPU numbers bit for bit as long as floating-point contraction is off on both
/// sides. The functions take the number type T of their arguments: the unit
/// tests run them in double against the CPU, the pattern building kernel in
/// Real. A change to the CPU formulas has to be made here as well.
namespace ActsExamples::CudaPatternMath {

/// @brief The number type of the pattern building kernel: the line, the
///        residual and its uncertainty, and the bending plane fit. The hit
///        payloads, the layer order and the output stay in double
using Real = float;

/// @brief A scalar argument that takes the number type of the other arguments
template <typename T>
using Scalar = std::type_identity_t<T>;

/// @brief Acts::s_epsilon
constexpr double epsilon = 3. * DBL_EPSILON;

/// @brief Acts::s_epsilon, in the number type T
template <typename T>
__host__ __device__ constexpr T epsilonOf() {
  if constexpr (std::is_same_v<T, float>) {
    return T{3} * FLT_EPSILON;
  } else {
    return T{3} * DBL_EPSILON;
  }
}
/// @brief The machine epsilon of the number type T
template <typename T>
__host__ __device__ constexpr T machineEpsilon() {
  if constexpr (std::is_same_v<T, float>) {
    return FLT_EPSILON;
  } else {
    return DBL_EPSILON;
  }
}
/// @brief The largest value of the number type T
template <typename T>
__host__ __device__ constexpr T largest() {
  if constexpr (std::is_same_v<T, float>) {
    return FLT_MAX;
  } else {
    return DBL_MAX;
  }
}

constexpr double pi = 3.14159265358979323846;

/// @brief Plain three-vector, usable in a kernel
template <typename T>
struct Vector3T {
  T x{0};
  T y{0};
  T z{0};
};
using Vector3 = Vector3T<double>;

/// @brief The same vector in another number type
template <typename To, typename From>
__host__ __device__ inline Vector3T<To> convert(const Vector3T<From>& a) {
  return {static_cast<To>(a.x), static_cast<To>(a.y), static_cast<To>(a.z)};
}

template <typename T>
__host__ __device__ inline Vector3T<T> operator+(const Vector3T<T>& a,
                                                 const Vector3T<T>& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
template <typename T>
__host__ __device__ inline Vector3T<T> operator-(const Vector3T<T>& a,
                                                 const Vector3T<T>& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
template <typename T>
__host__ __device__ inline Vector3T<T> operator*(Scalar<T> s,
                                                 const Vector3T<T>& a) {
  return {s * a.x, s * a.y, s * a.z};
}
template <typename T>
__host__ __device__ inline Vector3T<T> operator/(const Vector3T<T>& a,
                                                 Scalar<T> s) {
  return {a.x / s, a.y / s, a.z / s};
}
template <typename T>
__host__ __device__ inline T dot(const Vector3T<T>& a, const Vector3T<T>& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
template <typename T>
__host__ __device__ inline T norm(const Vector3T<T>& a) {
  return std::sqrt(dot(a, a));
}
template <typename T>
__host__ __device__ inline T square(T x) {
  return x * x;
}
/// @brief Acts::VectorHelpers::perp and Acts::fastHypot(x, y)
template <typename T>
__host__ __device__ inline T perp(const Vector3T<T>& a) {
  return std::sqrt(a.x * a.x + a.y * a.y);
}

/// @brief Acts::VectorHelpers::phi
template <typename T>
__host__ __device__ inline T phi(const Vector3T<T>& a) {
  return std::atan2(a.y, a.x);
}

/// @brief Acts::VectorHelpers::theta
template <typename T>
__host__ __device__ inline T theta(const Vector3T<T>& a) {
  return std::atan2(perp(a), a.z);
}

/// @brief Acts::detail::difference_periodic
template <typename T>
__host__ __device__ inline T differencePeriodic(T lhs, Scalar<T> rhs,
                                                Scalar<T> range) {
  T delta{std::fmod(lhs - rhs, range)};
  if ((T{2} * delta) < -range) {
    delta += range;
  } else if (range <= (T{2} * delta)) {
    delta -= range;
  }
  return delta;
}

/// @brief The quantities of one hit that the arithmetic reads
template <typename T>
struct HitT {
  Vector3T<T> position{};
  Vector3T<T> sensorDirection{};
  Vector3T<T> etaMeasurementDirection{};
  Vector3T<T> phiMeasurementDirection{};
  T localPositionY{0};
  T etaCovariance{0};
  T phiCovariance{0};
  T driftCovariance{0};
  std::uint32_t stationKey{0u};
  std::uint8_t station{0u};
  std::uint8_t locLayer{0u};
  std::uint8_t flags{0u};
};
using Hit = HitT<double>;

/// @brief Reads one row of the hit payload columns, in the number type T
template <typename T = double>
__host__ __device__ inline HitT<T> loadHit(const CudaHitPayloadArrays& hits,
                                           std::uint32_t row) {
  auto vector = [](double x, double y, double z) {
    return Vector3T<T>{static_cast<T>(x), static_cast<T>(y), static_cast<T>(z)};
  };
  HitT<T> hit{};
  hit.position =
      vector(hits.positionX[row], hits.positionY[row], hits.positionZ[row]);
  hit.sensorDirection =
      vector(hits.sensorDirectionX[row], hits.sensorDirectionY[row],
             hits.sensorDirectionZ[row]);
  hit.etaMeasurementDirection = vector(hits.etaMeasurementDirectionX[row],
                                       hits.etaMeasurementDirectionY[row],
                                       hits.etaMeasurementDirectionZ[row]);
  hit.phiMeasurementDirection = vector(hits.phiMeasurementDirectionX[row],
                                       hits.phiMeasurementDirectionY[row],
                                       hits.phiMeasurementDirectionZ[row]);
  hit.localPositionY = static_cast<T>(hits.localPositionY[row]);
  hit.etaCovariance = static_cast<T>(hits.etaCovariance[row]);
  hit.phiCovariance = static_cast<T>(hits.phiCovariance[row]);
  hit.driftCovariance = static_cast<T>(hits.driftCovariance[row]);
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
  hit.stationKey = row.stationKey;
  hit.station = row.station;
  hit.locLayer = row.locLayer;
  hit.flags = row.flags;
  return hit;
}

/// @brief PatternStateAux::intrinsicVariance: the hit covariance contracted
///        along a direction. A projected hit has its position along the
///        sensor taken by the projection, so the measurement along the sensor
///        drops out
template <typename T>
__host__ __device__ inline T intrinsicVariance(const HitT<T>& hit,
                                               const Vector3T<T>& c,
                                               bool isProjected) {
  if ((hit.flags & CudaHitFlags::straw) != 0u) {
    const T discVar{hit.driftCovariance};
    if (isProjected) {
      return discVar * dot(c, c);
    }
    const T vDotRsq{square(dot(hit.sensorDirection, c))};
    return discVar * (dot(c, c) - vDotRsq) + vDotRsq * hit.phiCovariance;
  }
  // A strip measures phi as well exactly when it has a second direction
  const bool hasSecond{(hit.flags & CudaHitFlags::measuresPhi) != 0u};
  const T bendTerm{hit.etaCovariance *
                   square(dot(hit.etaMeasurementDirection, c))};
  if (isProjected && !hasSecond) {
    return bendTerm;
  }
  return bendTerm +
         (hasSecond
              ? hit.phiCovariance * square(dot(hit.phiMeasurementDirection, c))
              : hit.phiCovariance * square(dot(hit.sensorDirection, c)));
}

/// @brief PatternStateAux::projToPhiPlane: the hit moved along its sensor
///        onto the bending plane, the plane of normal n through offset * n,
///        Acts::PlanarHelper::intersectPlane. A sensor parallel to the plane
///        has no intersection; the CPU then takes the position of an invalid
///        intersection, which is the origin, and so does this
template <typename T>
__host__ __device__ inline Vector3T<T> projectToPhiPlane(
    const HitT<T>& hit, const Vector3T<T>& n, Scalar<T> planeOffset) {
  const T offset{dot(planeOffset * n, n)};
  const T normDot{dot(n, hit.sensorDirection)};
  if (std::abs(normDot) < machineEpsilon<T>()) {
    return Vector3T<T>{};
  }
  const T path{(offset - dot(hit.position, n)) / normDot};
  return hit.position + path * hit.sensorDirection;
}

/// @brief The pattern line, as PatternStateAux holds it
template <typename T>
struct LineT {
  Vector3T<T> position{};
  Vector3T<T> direction{};
  T leverArm{0};
  bool useBeamspot{false};
};
using Line = LineT<double>;

/// @brief The beam spot, PatternStateAux::BeamspotInfo
template <typename T>
struct BeamspotT {
  Vector3T<T> position{};
  T radius{0};
  T length{0};
};
using Beamspot = BeamspotT<double>;

/// @brief The same beam spot in another number type
template <typename To, typename From>
__host__ __device__ inline BeamspotT<To> convert(const BeamspotT<From>& b) {
  return {convert<To>(b.position), static_cast<To>(b.radius),
          static_cast<To>(b.length)};
}

/// @brief PatternStateAux::updateLineParameters: the line through the
///        projected anchor and last hit, or through the beam spot when the two
///        are closer than minHitDistance
/// @param anchor The anchor hit projected onto the bending plane
/// @param last The last hit projected onto the bending plane
template <typename T>
__host__ __device__ inline LineT<T> lineThroughHits(
    const Vector3T<T>& anchor, const Vector3T<T>& last,
    Scalar<T> minHitDistance, const BeamspotT<T>& beamspot) {
  LineT<T> line{};
  Vector3T<T> d{last - anchor};
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

/// @brief The bending plane of a pattern: its phi, its offset from the origin
///        along the normal n = (-sin phi, cos phi, 0) and the covariance of
///        phi and offset, PatternStateAux::BendPlaneCov ePhiPhi, eSS, ePhiS
template <typename T>
struct BendPlaneT {
  T phi{0};
  T offset{0};
  Vector3T<T> normal{};
  T covPhiPhi{0};
  T covSS{0};
  T covPhiS{0};
};
using BendPlane = BendPlaneT<double>;

/// @brief Residual of a test hit against the pattern line, its uncertainty
///        and its degrees of freedom, PatternStateAux::LineTestRes
template <typename T>
struct LineResidualT {
  T residual{0};
  T sigma{0};
  std::uint8_t nDof{0u};
};
using LineResidual = LineResidualT<double>;

/// @brief PatternStateAux::computeLineResidual
/// @param line The pattern line
/// @param plane The bending plane of the pattern
/// @param nPhiLayers The number of phi layers of the pattern
/// @param anchor The line anchor hit
/// @param last The last inserted hit
/// @param test The hit tested against the line
/// @param beamspot The beam spot
template <typename T>
__host__ __device__ inline LineResidualT<T> lineResidual(
    const LineT<T>& line, const BendPlaneT<T>& plane, std::uint32_t nPhiLayers,
    const HitT<T>& anchor, const HitT<T>& last, const HitT<T>& test,
    const BeamspotT<T>& beamspot) {
  LineResidualT<T> res{};

  const bool projectTestHit{(test.flags & CudaHitFlags::measuresPhi) == 0u ||
                            nPhiLayers == 0u};
  const Vector3T<T> testPos{
      projectTestHit ? projectToPhiPlane(test, plane.normal, plane.offset)
                     : test.position};
  const Vector3T<T> K{testPos - line.position};
  const T KdotD{dot(K, line.direction)};

  const Vector3T<T> R{K - KdotD * line.direction};
  res.residual = norm(R);
  if (res.residual < epsilonOf<T>()) {
    res.residual = largest<T>();
    return res;
  }
  const Vector3T<T> resDir{R / res.residual};
  const T alpha{KdotD / line.leverArm};
  // Unit vector along the bending plane
  const T eRX{plane.normal.y};
  const T eRY{-plane.normal.x};

  T phiPlaneDerivativeAcc{0};
  T phiDisplDerivativeAcc{0};
  T residualCovAcc{0};

  auto covarianceTerm = [&](const HitT<T>& hit, const Vector3T<T>& pos,
                            const T preFactor, bool isProjected) {
    if (!isProjected) {
      residualCovAcc +=
          square(preFactor) * intrinsicVariance(hit, resDir, false);
      return;
    }
    const Vector3T<T>& sensorDir{hit.sensorDirection};
    const T projFactor{dot(sensorDir, resDir) / dot(sensorDir, plane.normal)};
    const Vector3T<T> trfDir{resDir - projFactor * plane.normal};

    residualCovAcc += square(preFactor) * intrinsicVariance(hit, trfDir, true);
    phiPlaneDerivativeAcc +=
        preFactor * projFactor * (pos.x * eRX + pos.y * eRY);
    phiDisplDerivativeAcc += preFactor * projFactor;
  };

  if (line.useBeamspot) {
    const T covS1 = square(beamspot.length) * square(resDir.z) +
                    square(beamspot.radius) * (1 - square(resDir.z));
    residualCovAcc += square(alpha - 1) * covS1;
  } else {
    covarianceTerm(anchor, line.position, alpha - 1, true);
  }

  const Vector3T<T> pos2{line.position + line.leverArm * line.direction};
  covarianceTerm(last, pos2, -alpha, true);

  covarianceTerm(test, testPos, T{1}, projectTestHit);

  const T projSigmaTerm{square(phiPlaneDerivativeAcc) * plane.covPhiPhi +
                        square(phiDisplDerivativeAcc) * plane.covSS +
                        2 * phiPlaneDerivativeAcc * phiDisplDerivativeAcc *
                            plane.covPhiS};

  res.sigma = std::sqrt(residualCovAcc + projSigmaTerm);
  res.nDof = projectTestHit ? 1u : 2u;
  return res;
}

/// @brief PatternStateAux::angleToBendPlane: the angle between a point and
///        the bending plane, as seen from the origin
template <typename T>
__host__ __device__ inline T angleToBendPlane(const BendPlaneT<T>& plane,
                                              const Vector3T<T>& pos) {
  const T dist{dot(plane.normal, pos) - plane.offset};
  const T rho{plane.normal.y * pos.x - plane.normal.x * pos.y};
  return std::atan2(dist, rho);
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
template <typename T = double>
__host__ __device__ inline T msSize(int msSector) {
  return static_cast<T>(msSector % 2 == 0 ? 0.4 * oneEighthOfPi
                                          : 0.6 * oneEighthOfPi);
}

/// @brief sectorMap::sectorWidth
template <typename T = double>
__host__ __device__ inline T msWidth(int msSector) {
  return msSize<T>(msSector) + static_cast<T>(overlap);
}

/// @brief sectorMap::sectorPhi
template <typename T = double>
__host__ __device__ inline T msPhi(int msSector) {
  const T piT{static_cast<T>(pi)};
  if (msSector < 10) {
    return piT * T(msSector - 1) / T{8};
  }
  return -piT * (T{2} - T(msSector - 1) / T{8});
}

/// @brief sectorMap::insideSector
template <typename T>
__host__ __device__ inline bool insideMsSector(int msSector, T phi) {
  const T piT{static_cast<T>(pi)};
  T dphi{phi - msPhi<T>(msSector)};
  if (dphi > piT) {
    dphi -= 2 * piT;
  }
  if (dphi < -piT) {
    dphi += 2 * piT;
  }
  return !(dphi < -msWidth<T>(msSector)) && !(dphi > msWidth<T>(msSector));
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
template <typename T = double>
__host__ __device__ inline T phi(int sector) {
  const int s1{msSector(sector)};
  const int s2{adjacentMsSector(sector)};
  if (s1 == s2) {
    return msPhi<T>(s1);
  }
  // The overlap starts at the upper edge of the first sector of the pair
  T phiO{msPhi<T>(s1) + msSize<T>(s1)};
  if (phiO > static_cast<T>(pi)) {
    phiO -= 2 * static_cast<T>(pi);
  }
  return phiO;
}

/// @brief ExpandedSector::sectorSize
template <typename T = double>
__host__ __device__ inline T size(int sector) {
  const int s1{msSector(sector)};
  if (s1 == adjacentMsSector(sector)) {
    return msSize<T>(s1);
  }
  return msWidth<T>(s1) - msSize<T>(s1);
}

/// @brief ExpandedSector::insideSector
template <typename T>
__host__ __device__ inline bool inside(int sector, T phi) {
  return insideMsSector(msSector(sector), phi) &&
         insideMsSector(adjacentMsSector(sector), phi);
}

/// @brief ExpandedSector::isNeighbour
__host__ __device__ inline bool isNeighbour(int sector1, int sector2) {
  const int dS{std::abs(sector2 - sector1)};
  return (dS < nExpanded - dS ? dS : nExpanded - dS) <= 1;
}
}  // namespace ExpandedSector

/// @brief PatternStateAux::updatePatternPhi: the bending plane from the phi
///        measuring hits of a pattern and the beam spot. Without phi hits it
///        is the centre of the sector, through the origin; with one, the line
///        through the hit and the beam spot; with more, the weighted straight
///        line fit of the hits and the beam spot in the transverse plane.
/// @param nPhiLayers The number of phi layers of the pattern
/// @param previousPhi The pattern phi so far, which fixes the fit weights
/// @param sector The expanded sector of the pattern
/// @param beamspot The beam spot
/// @param forEachPhiHit Calls its argument with every phi measuring hit of
///        the pattern, group by group, in the order they were added
/// @return The bending plane. A fit without weight or without a defined
///         direction, where the CPU throws, gives non-finite values
template <typename T, typename ForEachPhiHit>
__host__ __device__ inline BendPlaneT<T> fitBendPlane(
    std::uint32_t nPhiLayers, Scalar<T> previousPhi, int sector,
    const BeamspotT<T>& beamspot, ForEachPhiHit&& forEachPhiHit) {
  constexpr T halfPi{static_cast<T>(pi / 2.)};
  constexpr T piT{static_cast<T>(pi)};
  BendPlaneT<T> plane{};
  // The solution closest to the sector's direction
  auto alignPhiAndStore = [&](T newPatPhi) {
    const T refPhi{ExpandedSector::phi<T>(sector)};
    while (newPatPhi - refPhi > halfPi) {
      newPatPhi -= piT;
    }
    while (newPatPhi - refPhi < -halfPi) {
      newPatPhi += piT;
    }
    plane.phi = newPatPhi;
    plane.normal = Vector3T<T>{-std::sin(plane.phi), std::cos(plane.phi), 0};
  };
  if (nPhiLayers == 0u) {
    alignPhiAndStore(ExpandedSector::phi<T>(sector));
    plane.offset = 0;
    plane.covPhiPhi = square(ExpandedSector::size<T>(sector)) / T{3};
    plane.covSS = 0;
    plane.covPhiS = 0;
    return plane;
  }
  if (nPhiLayers == 1u) {
    // A copy: the visitor may hand over a hit that lives only for the call
    HitT<T> phiHit{};
    forEachPhiHit([&](const HitT<T>& hit) { phiHit = hit; });
    const T bsX{beamspot.position.x};
    const T bsY{beamspot.position.y};
    const T dX{phiHit.position.x - bsX};
    const T dY{phiHit.position.y - bsY};
    alignPhiAndStore(std::atan2(dY, dX));
    plane.offset = dot(plane.normal, beamspot.position);

    const T eRX{plane.normal.y};
    const T eRY{-plane.normal.x};
    const T lever{dX * eRX + dY * eRY};
    const T rhoBS{bsX * eRX + bsY * eRY};
    const T rhoHit{rhoBS + lever};

    const T varBS{square(beamspot.radius)};
    const T varHit{intrinsicVariance(phiHit, plane.normal, false)};
    const T invL2{T{1} / square(lever)};

    plane.covPhiPhi = (varHit + varBS) * invL2;
    plane.covPhiS = -(rhoBS * varHit + rhoHit * varBS) * invL2;
    plane.covSS = (square(rhoBS) * varHit + square(rhoHit) * varBS) * invL2;
    return plane;
  }

  const Vector3T<T> n0{-std::sin(previousPhi), std::cos(previousPhi), 0};
  const T bsX{beamspot.position.x};
  const T bsY{beamspot.position.y};
  T weightSum{0};
  T Sxx{0};
  T Sxy{0};
  T Syy{0};
  T centroidX{0};
  T centroidY{0};
  forEachPhiHit([&](const HitT<T>& hit) {
    const T variance{intrinsicVariance(hit, n0, false)};
    if (variance < epsilonOf<T>()) {
      return;
    }
    const T weight{T{1} / variance};
    const T dX{hit.position.x - bsX};
    const T dY{hit.position.y - bsY};
    const T wdX{weight * dX};
    const T wdY{weight * dY};

    weightSum += weight;
    centroidX += wdX;
    centroidY += wdY;
    Sxx += wdX * dX;
    Sxy += wdX * dY;
    Syy += wdY * dY;
  });
  weightSum += T{1} / square(beamspot.radius);

  const T weightSumInv{T{1} / weightSum};
  centroidX *= weightSumInv;
  centroidY *= weightSumInv;

  const T Mxx{Sxx - weightSum * square(centroidX)};
  const T Mxy{Sxy - weightSum * centroidX * centroidY};
  const T Myy{Syy - weightSum * square(centroidY)};

  const T eigenvalSeparation{std::sqrt(square(Mxx - Myy) + square(T{2} * Mxy))};
  const T eigenValSeparationInv{T{1} / eigenvalSeparation};

  alignPhiAndStore(T{0.5} * std::atan2(T{2} * Mxy, Mxx - Myy));

  centroidX += bsX;
  centroidY += bsY;
  plane.offset = plane.normal.x * centroidX + plane.normal.y * centroidY;
  const T eRX{plane.normal.y};
  const T eRY{-plane.normal.x};
  const T rhoCentroid{centroidX * eRX + centroidY * eRY};

  plane.covPhiPhi = eigenValSeparationInv;
  plane.covPhiS = -rhoCentroid * eigenValSeparationInv;
  plane.covSS = square(rhoCentroid) * eigenValSeparationInv + weightSumInv;
  return plane;
}

}  // namespace ActsExamples::CudaPatternMath

namespace ActsExamples {

/// @brief The constants of the GlobalPatternFinder config that the kernels
///        read
struct CudaPatternFinderConfig {
  double nResidualSigma{3.};
  /// @brief The residual cut of a hit with two degrees of freedom, as
  ///        GlobalPatternFinder computes it from nResidualSigma
  double cut2Dof{0.};
  double phiTolerance{3. * CudaPatternMath::pi / 180.};
  double loosenAcceptanceFactor{2.};
  double minHitDistance4Line{40.};
  double maxThetaSizeOverlap{5. * CudaPatternMath::pi / 180.};
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

/// @brief The device memory of the pattern finding. The finding runs once
///        per event; what does not fit is chosen by priority and reported,
///        never built again.
struct CudaPatternFinderSizes {
  /// @brief Device memory the seeds of an event take from as they need it,
  ///        and the event's patterns after them
  std::size_t workspaceBytes{std::size_t{1} << 30u};
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
  /// @brief Device: seed preparation, builds and accepts
  double kernelsUs{0.};
  /// @brief Device, part of kernelsUs: the seed checks
  double seedsUs{0.};
  /// @brief Device, part of kernelsUs: the building of every seed
  double buildUs{0.};
  /// @brief Device, part of kernelsUs: the event's patterns gathered, their
  ///        overlaps resolved and the kept ones compacted
  double overlapsUs{0.};
  /// @brief Device: copy of the event's patterns to the host
  double downloadUs{0.};
  /// @brief Host: phi-only hits and the conversion of the patterns
  double tailUs{0.};
  std::size_t nSeeds{0u};
  std::size_t nCandidates{0u};
  /// @brief Patterns in eta accepted on the device
  std::size_t nPatternsInEta{0u};
  /// @brief The most patterns one seed held at once
  std::size_t largestBeam{0u};
  /// @brief CudaPatternOverflow bits of what did not fit; zero if everything
  ///        fitted
  std::uint8_t overflow{0u};
};

/// @brief Copies the constants out of the CPU config
/// @tparam Config ActsExamples::GlobalPatternFinder_t::Config
/// @tparam Beamspot ActsExamples::GlobalPatternFinder_t::BeamspotInfo
template <typename Config, typename Beamspot>
CudaPatternFinderConfig makeCudaPatternFinderConfig(const Config& cfg,
                                                    const Beamspot& beamspot) {
  return CudaPatternFinderConfig{
      .nResidualSigma = cfg.nResidualSigma,
      .cut2Dof = std::sqrt(
          -2. * std::log(std::erfc(cfg.nResidualSigma / std::numbers::sqrt2))),
      .phiTolerance = cfg.phiTolerance,
      .loosenAcceptanceFactor = cfg.loosenAcceptanceFactor,
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

/// @brief Runs the candidate checks of the CPU seed loop on the device, a
///        warp per seed: the candidate count, the group count as a
///        reduction, the sort by layer as a sorting network and the global
///        layer numbers as a prefix sum, with their cuts.
///
/// The status, the global layers and the position of the seed hit in its
/// sorted list are written to the output columns of the lists; the position is
/// cudaInvalidHitPosition if the seed hit is not in its own list. Both
/// containers must be on the device. The call waits only for the stream.
void prepareCudaSeeds(const CudaHitPayloadContainer& hits,
                      CudaCandidateListContainer& lists,
                      const CudaPatternFinderConfig& cfg, cudaStream_t stream);

/// @brief The measurement surfaces of the tracking geometry on the device,
///        sorted by geometry identifier: the rotation of surface s, column
///        major, is rotation[9 * s, 9 * s + 9). The arrays are not owned.
struct CudaSurfaceTableArrays {
  const std::uint64_t* geometryId{nullptr};
  const double* rotation{nullptr};
  std::uint32_t nSurfaces{0u};
};

/// @brief The positions of the hits the pattern finding uses, which the host
///        computes for the search tree, and the rotation of each bucket into
///        the global frame, column major, rotation[9 * b, 9 * b + 9). Row r is
///        space point r of the CudaMuonSpacePointContainer, which holds these
///        hits only; its bucket ranges number the buckets. The arrays are not
///        owned.
struct CudaUsedHitArrays {
  const double* positionX{nullptr};
  const double* positionY{nullptr};
  const double* positionZ{nullptr};
  const double* bucketRotation{nullptr};
  std::uint32_t nHits{0u};
  std::uint32_t nBuckets{0u};
};

/// @brief Builds the hit payloads on the device, one thread per hit, from the
///        space points (CudaMuonSpacePointArrays) and the surfaces: the
///        quantities of makeCudaHitPayloadRow, with the formulas of HitPayload.
/// @param spacePoints Device arrays of a CudaMuonSpacePointContainer
/// @param used The hits to build, row r of the output from row r of used
/// @param surfaces The measurement surfaces
/// @param payloads Device arrays of the output, with used.nHits rows
/// @param nMissingSurfaces Device counter of the hits whose surface is not in
///        the table, which the CPU rejects with an exception
/// @param stream The stream; the call does not wait
void buildCudaHitPayloads(const CudaMuonSpacePointArrays& spacePoints,
                          const CudaUsedHitArrays& used,
                          const CudaSurfaceTableArrays& surfaces,
                          const CudaHitPayloadArrays& payloads,
                          std::uint32_t* nMissingSurfaces, cudaStream_t stream);

/// @brief Finds the patterns in eta of an event on the device, in one pass.
///
/// The seed checks run a warp per seed: the group count as a reduction, the
/// sort by layer as a sorting network and the layer numbers as a prefix sum.
/// Every seed is then built by a warp, all seeds at once, one layer at a
/// time: every hit of a layer is tested against every pattern at once, with
/// the line the pattern had before the layer, and each compatible hit gives a
/// pattern of its own. The backward passes of all forward patterns run side
/// by side. The overlaps, inside a seed and among all the patterns of the
/// event, are resolved by priority (more good groups, more bending layers,
/// more precision layers, smaller residual, then the index) rather than in
/// seed order, and the skip rule is not applied.
///
/// The seeds take their memory from one arena of sizes.workspaceBytes as
/// they need it; a pattern's hit slots are sized by the layers of its seed's
/// list, so they cannot overflow. When the arena runs short at a layer, the
/// seed keeps the best of that layer's patterns that fit, by the same
/// priority, and the patterns overflow bit is set. The event's patterns are
/// sized by what the seeds produced. Only the kept patterns are copied back:
/// the returned records hold them alone, in seed order, as the slots of entry
/// 0. The hits and the lists must be on the device; the call synchronizes the
/// stream. The counts and the kept patterns come back through `readback`,
/// page-locked memory the caller keeps between events, or through a buffer
/// of the call's own if none is given.
CudaPatternRecordContainer findCudaPatternsInEta(
    const CudaHitPayloadContainer& hits, CudaCandidateListContainer& lists,
    const CudaPatternFinderConfig& cfg, const CudaPatternFinderSizes& sizes,
    cudaStream_t stream, CudaPatternFinderTiming* timing = nullptr,
    CudaPinnedBuffer* readback = nullptr);

}  // namespace ActsExamples

// The driver works with the CPU types of the pattern finder, which the device
// code does not need
#ifndef __CUDACC__
#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "ActsExamples/TrackFinding/GlobalPatternFinderAlgorithm.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ActsExamples {

/// @brief GlobalPatternFinder_t::findPatterns on the GPU. The patterns in eta
///        are found by findCudaPatternsInEta; the phi-only hits are added and
///        the patterns converted on the host, as the CPU does it.
class CudaGlobalPatternFinder {
 public:
  using OutputPattern = GlobalPatternFinder_t::OutputPattern;
  using PatternStateAux = GlobalPatternFinder_t::PatternStateAux;
  using BeamspotInfo = GlobalPatternFinder_t::BeamspotInfo;

  struct Config {
    /// @brief The configuration of the CPU pattern finder
    GlobalPatternFinder_t::Config finder{};
    /// @brief The sizes of the device buffers
    CudaPatternFinderSizes sizes{};
    /// @brief The tracking geometry, whose surfaces go to the device once,
    ///        and whether MDT hits take part
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry{};
    bool useMdtHits{true};
    /// @brief The share of the device's free memory the pattern building
    ///        takes, measured at the first event. Zero takes
    ///        sizes.workspaceBytes instead
    double workspaceFraction{0.5};
  };

  explicit CudaGlobalPatternFinder(
      Config cfg,
      std::unique_ptr<const Acts::Logger> logger = Acts::getDefaultLogger(
          "CudaGlobalPatternFinder", Acts::Logging::INFO));

  /// @brief Finds the patterns of an event, converted to the output of the
  ///        algorithm. The host computes
  ///        the global positions, the search tree and the candidate lists;
  ///        the space points go to the device, which builds the hit payloads
  ///        and finds the patterns in eta. The host builds the payloads of the
  ///        hits of the kept patterns only, for the phi-only hits and the
  ///        conversion.
  /// @param gctx Geometry context
  /// @param spacePoints The space points of the event
  /// @param seedSelector Selector of the seed hits
  /// @param onlyPhiProvider Provider of the phi-only hits of a pattern
  /// @param beamspot The beam spot
  /// @param timing Filled with the time of each stage, if given
  MuonGlobalPatternContainer findPatterns(
      const Acts::GeometryContext& gctx,
      const MuonSpacePointContainer& spacePoints,
      const SeedSelector& seedSelector,
      const OnlyPhiHitsProvider& onlyPhiProvider, const BeamspotInfo& beamspot,
      CudaPatternFinderTiming* timing = nullptr) const;

  ~CudaGlobalPatternFinder();

 private:
  /// @brief The hit payload of a row of the device hit columns
  using PayloadOf = std::function<const HitPayload&(std::uint32_t row)>;

  /// @brief A pattern found on the device as the CPU's PatternStateAux
  PatternStateAux makePatternState(const Acts::GeometryContext& gctx,
                                   const PayloadOf& payloadOf,
                                   const CudaPatternRecordContainer& event,
                                   std::size_t slot,
                                   const BeamspotInfo& beamspot) const;

  /// @brief The patterns of status ok of entry 0 of the records, with their
  ///        phi-only hits, converted to the output
  std::vector<OutputPattern> finishPatterns(
      const Acts::GeometryContext& gctx, const PayloadOf& payloadOf,
      const CudaPatternRecordContainer& event,
      const OnlyPhiHitsProvider& onlyPhiProvider,
      const BeamspotInfo& beamspot) const;

  /// @brief The measurement surfaces on the device, filled at the first event
  struct SurfaceTable {
    std::uint64_t* geometryId{nullptr};
    double* rotation{nullptr};
    std::uint32_t nSurfaces{0u};
  };
  const SurfaceTable& surfaceTable(const Acts::GeometryContext& gctx) const;
  mutable SurfaceTable m_surfaces{};
  mutable std::once_flag m_surfacesOnce{};
  /// @brief The bytes of the pattern building, from workspaceFraction; zero
  ///        takes sizes.workspaceBytes
  mutable std::size_t m_arenaBytes{0u};

  /// @brief The stream and the transfer memory of one event, kept between
  ///        events; one per event in flight
  struct Workspace;
  std::unique_ptr<Workspace> acquireWorkspace() const;
  void releaseWorkspace(std::unique_ptr<Workspace> workspace) const;
  /// @brief Gives the workspace back when the event is done
  struct WorkspaceReturn {
    const CudaGlobalPatternFinder& finder;
    std::unique_ptr<Workspace>& workspace;
    ~WorkspaceReturn();
  };
  mutable std::mutex m_workspaceMutex{};
  mutable std::vector<std::unique_ptr<Workspace>> m_workspaces{};

  /// @brief GlobalPatternFinder::addPhiOnlyHits
  void addPhiOnlyHits(const Acts::GeometryContext& gctx,
                      const BeamspotInfo& beamspot,
                      const OnlyPhiHitsProvider& onlyPhiProvider,
                      std::vector<PatternStateAux>& patterns) const;

  const Acts::Logger& logger() const { return *m_logger; }

  Config m_cfg;
  std::unique_ptr<const Acts::Logger> m_logger;
};

/// @brief GlobalPatternFinderAlgorithm::convertToPattern: the space points of
///        the patterns, which point into the payloads the patterns were found
///        with, so it is called while those are alive
MuonGlobalPatternContainer toMuonGlobalPatterns(
    const std::vector<GlobalPatternFinder_t::OutputPattern>& found);

/// @brief GlobalPatternFinderAlgorithm with the pattern finding on the GPU.
///        It reads the same space points and writes the same patterns; the
///        search tree, the phi-only hits and the conversion of the patterns
///        stay on the host.
class CudaGlobalPatternFinderAlgorithm final : public IAlgorithm {
 public:
  struct Config : GlobalPatternFinderAlgorithm::Config {
    /// @brief Device memory of the pattern building, used when
    ///        workspaceFraction is zero
    std::size_t workspaceBytes{std::size_t{1} << 30u};
    /// @brief Share of the device's free memory for the pattern building,
    ///        see CudaGlobalPatternFinder::Config
    double workspaceFraction{0.5};
    /// @brief Key of the CudaPatternFinderTiming of each event in the event
    ///        store; nothing is written if empty
    std::string outTiming{};
  };

  CudaGlobalPatternFinderAlgorithm(const Config& cfg,
                                   std::unique_ptr<const Acts::Logger> logger);

  ProcessCode execute(const AlgorithmContext& ctx) const override;

  const Config& config() const { return m_cfg; }

 private:
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
