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

/// @brief Arithmetic of the global pattern finder for the host and the device
namespace ActsExamples::CudaPatternMath {

/// Floating-point type of the pattern building kernel; hit payloads, layer
/// sorting and output use double
using Real = float;

template <typename T>
using Scalar = std::type_identity_t<T>;

constexpr double epsilon = 3. * DBL_EPSILON;

template <typename T>
__host__ __device__ constexpr T epsilonOf() {
  if constexpr (std::is_same_v<T, float>) {
    return T{3} * FLT_EPSILON;
  } else {
    return T{3} * DBL_EPSILON;
  }
}
template <typename T>
__host__ __device__ constexpr T machineEpsilon() {
  if constexpr (std::is_same_v<T, float>) {
    return FLT_EPSILON;
  } else {
    return DBL_EPSILON;
  }
}
template <typename T>
__host__ __device__ constexpr T largest() {
  if constexpr (std::is_same_v<T, float>) {
    return FLT_MAX;
  } else {
    return DBL_MAX;
  }
}

constexpr double pi = 3.14159265358979323846;

template <typename T>
struct Vector3T {
  T x{0};
  T y{0};
  T z{0};
};
using Vector3 = Vector3T<double>;

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
template <typename T>
__host__ __device__ inline T perp(const Vector3T<T>& a) {
  return std::sqrt(a.x * a.x + a.y * a.y);
}

template <typename T>
__host__ __device__ inline T phi(const Vector3T<T>& a) {
  return std::atan2(a.y, a.x);
}

template <typename T>
__host__ __device__ inline T theta(const Vector3T<T>& a) {
  return std::atan2(perp(a), a.z);
}

/// lhs - rhs, mapped to [-range/2, range/2)
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

/// Variance of a hit's position along c. For a projected hit, the
/// measurement along the sensor does not contribute.
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

/// Moves a hit along its sensor onto the plane with normal n through
/// offset * n; the origin if the sensor is parallel to the plane
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

template <typename T>
struct LineT {
  Vector3T<T> position{};
  Vector3T<T> direction{};
  T leverArm{0};
  bool useBeamspot{false};
};
using Line = LineT<double>;

template <typename T>
struct BeamspotT {
  Vector3T<T> position{};
  T radius{0};
  T length{0};
};
using Beamspot = BeamspotT<double>;

template <typename To, typename From>
__host__ __device__ inline BeamspotT<To> convert(const BeamspotT<From>& b) {
  return {convert<To>(b.position), static_cast<To>(b.radius),
          static_cast<To>(b.length)};
}

/// Through the beam spot instead of the anchor when the two hits are closer
/// than minHitDistance
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

/// Bending plane: phi, offset from the origin along the normal
/// n = (-sin phi, cos phi, 0), and the covariance of phi and offset
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

template <typename T>
struct LineResidualT {
  T residual{0};
  T sigma{0};
  std::uint8_t nDof{0u};
};
using LineResidual = LineResidualT<double>;

/// Residual of a test hit against the pattern line, with its uncertainty
/// from the anchor, last and test hits and the bending plane
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

/// Seen from the origin
template <typename T>
__host__ __device__ inline T angleToBendPlane(const BendPlaneT<T>& plane,
                                              const Vector3T<T>& pos) {
  const T dist{dot(plane.normal, pos) - plane.offset};
  const T rho{plane.normal.y * pos.x - plane.normal.x * pos.y};
  return std::atan2(dist, rho);
}

/// Numbered as in Muon::MuonStationIndex
namespace Station {
constexpr std::uint8_t BI = 0u, BM = 1u, BO = 2u, BE = 3u, EI = 4u, EM = 5u,
                       EO = 6u, EE = 7u;
constexpr std::uint8_t nGroups = 8u;
}  // namespace Station
namespace Layer {
constexpr std::uint8_t Inner = 0u, Middle = 1u, Outer = 2u, Extended = 3u,
                       BarrelExtended = 4u;
}

__host__ __device__ inline bool isBarrel(std::uint8_t station) {
  return station <= Station::BE;
}

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

__host__ __device__ inline std::uint8_t groupIndex(const Hit& hit) {
  return hit.station;
}

/// A BO and an EO hit, not expected in one pattern, compare false
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

/// The 16 muon sectors and the 16 overlaps between neighbours, numbered 0 to
/// 31: even numbers are sector centres, odd ones the overlap with the next
namespace ExpandedSector {
constexpr int nMsSectors = 16;
constexpr int nExpanded = 2 * nMsSectors;
constexpr double oneEighthOfPi = pi / 8.;
constexpr double overlap = 0.1 * oneEighthOfPi;

/// Half width without overlap
template <typename T = double>
__host__ __device__ inline T msSize(int msSector) {
  return static_cast<T>(msSector % 2 == 0 ? 0.4 * oneEighthOfPi
                                          : 0.6 * oneEighthOfPi);
}

/// Half width with overlap
template <typename T = double>
__host__ __device__ inline T msWidth(int msSector) {
  return msSize<T>(msSector) + static_cast<T>(overlap);
}

template <typename T = double>
__host__ __device__ inline T msPhi(int msSector) {
  const T piT{static_cast<T>(pi)};
  if (msSector < 10) {
    return piT * T(msSector - 1) / T{8};
  }
  return -piT * (T{2} - T(msSector - 1) / T{8});
}

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

__host__ __device__ inline int msSector(int sector) {
  return sector < 2 ? nMsSectors : sector / 2;
}

/// The other sector of an overlap, or the sector itself for a centre
__host__ __device__ inline int adjacentMsSector(int sector) {
  const int ms{msSector(sector)};
  if (sector % 2 == 0) {
    return ms;
  }
  return ms == nMsSectors ? 1 : ms + 1;
}

template <typename T = double>
__host__ __device__ inline T phi(int sector) {
  const int s1{msSector(sector)};
  const int s2{adjacentMsSector(sector)};
  if (s1 == s2) {
    return msPhi<T>(s1);
  }
  // An overlap is centred at the upper edge of its first sector
  T phiO{msPhi<T>(s1) + msSize<T>(s1)};
  if (phiO > static_cast<T>(pi)) {
    phiO -= 2 * static_cast<T>(pi);
  }
  return phiO;
}

template <typename T = double>
__host__ __device__ inline T size(int sector) {
  const int s1{msSector(sector)};
  if (s1 == adjacentMsSector(sector)) {
    return msSize<T>(s1);
  }
  return msWidth<T>(s1) - msSize<T>(s1);
}

template <typename T>
__host__ __device__ inline bool inside(int sector, T phi) {
  return insideMsSector(msSector(sector), phi) &&
         insideMsSector(adjacentMsSector(sector), phi);
}

__host__ __device__ inline bool isNeighbour(int sector1, int sector2) {
  const int dS{std::abs(sector2 - sector1)};
  return (dS < nExpanded - dS ? dS : nExpanded - dS) <= 1;
}
}  // namespace ExpandedSector

/// Fits the bending plane to the phi hits of a pattern, visited by
/// forEachPhiHit, and the beam spot. No phi hit uses the sector centre.
/// One or more phi hits share one regression; weights are frozen at
/// previousPhi, which is the current plane.
template <typename T, typename ForEachPhiHit>
__host__ __device__ inline BendPlaneT<T> fitBendPlane(
    std::uint32_t nPhiLayers, Scalar<T> previousPhi, int sector,
    const BeamspotT<T>& beamspot, ForEachPhiHit&& forEachPhiHit) {
  constexpr T halfPi{static_cast<T>(pi / 2.)};
  constexpr T piT{static_cast<T>(pi)};
  BendPlaneT<T> plane{};
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

/// @brief Pattern finder configuration read by the kernels
struct CudaPatternFinderConfig {
  double nResidualSigma{3.};
  /// Residual cut for two degrees of freedom, from nResidualSigma
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

/// @brief Device memory of the pattern finding. If the patterns do not fit,
///        the best ones are kept and the overflow is reported.
struct CudaPatternFinderSizes {
  /// Arena for the patterns of one event
  std::size_t workspaceBytes{std::size_t{1} << 30u};
};

/// @brief Time per stage of one event in microseconds: wall time on the
///        host, CUDA events on the device
struct CudaPatternFinderTiming {
  double hostPreUs{0.};
  double uploadUs{0.};
  double kernelsUs{0.};
  double downloadUs{0.};
  /// Phi-only hits and conversion of the patterns
  double hostPostUs{0.};
  std::size_t nSeeds{0u};
  std::size_t nCandidates{0u};
  std::size_t nPatternsInEta{0u};
  /// CudaPatternOverflow bits
  std::uint8_t overflow{0u};
};

/// @brief Kernel configuration from the pattern finder configuration
/// @tparam Config GlobalPatternFinder_t::Config
/// @tparam Beamspot GlobalPatternFinder_t::BeamspotInfo
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

/// @brief Seed checks on the device: group counts, sorting by layer and
///        global layer numbers, written to the lists. Asynchronous.
void prepareCudaSeeds(const CudaHitPayloadContainer& hits,
                      CudaCandidateListContainer& lists,
                      const CudaPatternFinderConfig& cfg, cudaStream_t stream);

/// @brief Measurement surfaces on the device, sorted by geometry identifier;
///        the column-major rotation of surface s is rotation[9 * s, 9 * s + 9)
struct CudaSurfaceTableArrays {
  const std::uint64_t* geometryId{nullptr};
  const double* rotation{nullptr};
  std::uint32_t nSurfaces{0u};
};

/// @brief Global positions of the used hits and the column-major rotation of
///        each bucket into the global frame, rotation[9 * b, 9 * b + 9). Row r
///        is space point r of the CudaMuonSpacePointContainer.
struct CudaUsedHitArrays {
  const double* positionX{nullptr};
  const double* positionY{nullptr};
  const double* positionZ{nullptr};
  const double* bucketRotation{nullptr};
  std::uint32_t nHits{0u};
  std::uint32_t nBuckets{0u};
};

/// @brief Builds the hit payloads on the device. Asynchronous.
/// @param nMissingSurfaces Device counter of hits whose surface is not in
///        the table
void buildCudaHitPayloads(const CudaMuonSpacePointArrays& spacePoints,
                          const CudaUsedHitArrays& used,
                          const CudaSurfaceTableArrays& surfaces,
                          const CudaHitPayloadArrays& payloads,
                          std::uint32_t* nMissingSurfaces, cudaStream_t stream);

/// @brief Seed selection for the device search
struct CudaSeedSelection {
  /// Bit l set: hits of LayerIndex l seed
  std::uint32_t seedingLayers{0u};
  bool seedFromMdt{false};
  double thetaSearchWindow{0.};
};

/// @brief Bucket transforms into the global frame, column-major
///        rotation[9 * b, 9 * b + 9) and translation[3 * b, 3 * b + 3), and
///        the output global positions
struct CudaSearchArrays {
  const double* bucketRotation{nullptr};
  const double* bucketTranslation{nullptr};
  double* positionX{nullptr};
  double* positionY{nullptr};
  double* positionZ{nullptr};
  std::uint32_t nHits{0u};
  std::uint32_t nBuckets{0u};
};

/// @brief Finds the candidates of every seed on the device. Waits for the
///        numbers of seeds and candidates, read through readback.
void findCudaCandidates(const CudaMuonSpacePointArrays& spacePoints,
                        const CudaSearchArrays& search,
                        const CudaSeedSelection& selection,
                        CudaCandidateListContainer& lists,
                        CudaPinnedBuffer& readback, cudaStream_t stream);

/// @brief One-off setup of the device: loads every kernel, keeps freed memory
///        in the device's memory pool and fills the pool with the arena, so
///        that no event pays for them. Synchronises the stream.
void prepareCudaPatternFinder(std::size_t arenaBytes, cudaStream_t stream);

/// @brief Finds the patterns in eta of an event on the device: one warp per
///        seed, then overlap resolution over the event by priority. If the
///        arena runs short, the best patterns are kept and the overflow bit
///        is set. Synchronises the stream.
/// @param readback Page-locked memory reused between events, if given
/// @return The kept patterns, as the slots of entry 0
CudaPatternRecordContainer findCudaPatternsInEta(
    const CudaHitPayloadContainer& hits, CudaCandidateListContainer& lists,
    const CudaPatternFinderConfig& cfg, const CudaPatternFinderSizes& sizes,
    cudaStream_t stream, CudaPatternFinderTiming* timing = nullptr,
    CudaPinnedBuffer* readback = nullptr);

}  // namespace ActsExamples

// Host-only part: depends on the host types of the pattern finder
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

/// @brief Global pattern finder with the patterns in eta found on the GPU;
///        the search tree, phi-only hits and conversion run on the host
class CudaGlobalPatternFinder {
 public:
  using OutputPattern = GlobalPatternFinder_t::OutputPattern;
  using PatternStateAux = GlobalPatternFinder_t::PatternStateAux;
  using BeamspotInfo = GlobalPatternFinder_t::BeamspotInfo;

  struct Config {
    GlobalPatternFinder_t::Config finder{};
    CudaPatternFinderSizes sizes{};
    /// Its surfaces are uploaded at setup
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry{};
    bool useMdtHits{true};
    /// Fraction of the free device memory used for the arena; zero uses
    /// sizes.workspaceBytes
    double workspaceFraction{0.5};
    /// Reserved at setup for each transfer buffer of the event
    std::size_t transferBytes{std::size_t{16} << 20u};
  };

  explicit CudaGlobalPatternFinder(
      Config cfg,
      std::unique_ptr<const Acts::Logger> logger = Acts::getDefaultLogger(
          "CudaGlobalPatternFinder", Acts::Logging::INFO));

  /// @brief One-off setup: CUDA context, surface table, arena size, kernels,
  ///        memory pool and transfer buffers. Runs once; findPatterns calls it
  ///        if it has not run.
  void initialize(const Acts::GeometryContext& gctx) const;

  /// @brief Finds the patterns of an event
  /// @param timing Filled with the time of each stage, if given
  MuonGlobalPatternContainer findPatterns(
      const Acts::GeometryContext& gctx,
      const MuonSpacePointContainer& spacePoints,
      const SeedSelector& seedSelector,
      const OnlyPhiHitsProvider& onlyPhiProvider, const BeamspotInfo& beamspot,
      CudaPatternFinderTiming* timing = nullptr) const;

  ~CudaGlobalPatternFinder();

 private:
  using PayloadOf = std::function<const HitPayload&(std::uint32_t row)>;

  PatternStateAux makePatternState(const Acts::GeometryContext& gctx,
                                   const PayloadOf& payloadOf,
                                   const CudaPatternRecordContainer& event,
                                   std::size_t slot,
                                   const BeamspotInfo& beamspot) const;

  std::vector<OutputPattern> finishPatterns(
      const Acts::GeometryContext& gctx, const PayloadOf& payloadOf,
      const CudaPatternRecordContainer& event,
      const OnlyPhiHitsProvider& onlyPhiProvider,
      const BeamspotInfo& beamspot) const;

  struct SurfaceTable {
    std::uint64_t* geometryId{nullptr};
    double* rotation{nullptr};
    std::uint32_t nSurfaces{0u};
  };
  const SurfaceTable& surfaceTable(const Acts::GeometryContext& gctx) const;
  mutable SurfaceTable m_surfaces{};
  mutable std::once_flag m_surfacesOnce{};
  mutable std::once_flag m_setupOnce{};
  mutable std::size_t m_arenaBytes{0u};

  /// Stream and transfer buffers, one per event in flight
  struct Workspace;
  std::unique_ptr<Workspace> acquireWorkspace() const;
  void releaseWorkspace(std::unique_ptr<Workspace> workspace) const;
  struct WorkspaceReturn {
    const CudaGlobalPatternFinder& finder;
    std::unique_ptr<Workspace>& workspace;
    ~WorkspaceReturn();
  };
  mutable std::mutex m_workspaceMutex{};
  mutable std::vector<std::unique_ptr<Workspace>> m_workspaces{};

  void addPhiOnlyHits(const Acts::GeometryContext& gctx,
                      const BeamspotInfo& beamspot,
                      const OnlyPhiHitsProvider& onlyPhiProvider,
                      std::vector<PatternStateAux>& patterns) const;

  const Acts::Logger& logger() const { return *m_logger; }

  Config m_cfg;
  std::unique_ptr<const Acts::Logger> m_logger;
};

/// @brief The patterns point into their hit payloads, which must be alive
MuonGlobalPatternContainer toMuonGlobalPatterns(
    const std::vector<GlobalPatternFinder_t::OutputPattern>& found);

/// @brief GlobalPatternFinderAlgorithm with the patterns in eta found on the
///        GPU
class CudaGlobalPatternFinderAlgorithm final : public IAlgorithm {
 public:
  struct Config : GlobalPatternFinderAlgorithm::Config {
    /// Arena size when workspaceFraction is zero
    std::size_t workspaceBytes{std::size_t{1} << 30u};
    /// See CudaGlobalPatternFinder::Config
    double workspaceFraction{0.5};
    /// Event store key of the timing; not written if empty
    std::string outTiming{};
  };

  CudaGlobalPatternFinderAlgorithm(const Config& cfg,
                                   std::unique_ptr<const Acts::Logger> logger);

  /// @brief Sets up the device with the default geometry context
  ProcessCode initialize() override;

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
