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
__host__ __device__ inline double intrinsicVariance(const Hit& hit,
                                                    const Vector3& c,
                                                    bool isProjected) {
  if ((hit.flags & CudaHitFlags::straw) != 0u) {
    const double discVar{hit.driftCovariance};
    if (isProjected) {
      return discVar * dot(c, c);
    }
    const double vDotRsq{square(dot(hit.sensorDirection, c))};
    return discVar * (dot(c, c) - vDotRsq) + vDotRsq * hit.phiCovariance;
  }
  // A strip measures phi as well exactly when it has a second direction
  const bool hasSecond{(hit.flags & CudaHitFlags::measuresPhi) != 0u};
  const double bendTerm{hit.etaCovariance *
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
__host__ __device__ inline Vector3 projectToPhiPlane(const Hit& hit,
                                                     const Vector3& n,
                                                     double planeOffset) {
  const double offset{dot(planeOffset * n, n)};
  const double normDot{dot(n, hit.sensorDirection)};
  if (std::abs(normDot) < DBL_EPSILON) {
    return Vector3{};
  }
  const double path{(offset - dot(hit.position, n)) / normDot};
  return hit.position + path * hit.sensorDirection;
}

/// @brief The pattern line, as PatternStateAux holds it
struct Line {
  Vector3 position{};
  Vector3 direction{};
  double leverArm{0.};
  bool useBeamspot{false};
};

/// @brief The beam spot, PatternStateAux::BeamspotInfo
struct Beamspot {
  Vector3 position{};
  double radius{0.};
  double length{0.};
};

/// @brief PatternStateAux::updateLineParameters: the line through the
///        projected anchor and last hit, or through the beam spot when the two
///        are closer than minHitDistance
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

/// @brief The bending plane of a pattern: its phi, its offset from the origin
///        along the normal n = (-sin phi, cos phi, 0) and the covariance of
///        phi and offset, PatternStateAux::BendPlaneCov ePhiPhi, eSS, ePhiS
struct BendPlane {
  double phi{0.};
  double offset{0.};
  Vector3 normal{};
  double covPhiPhi{0.};
  double covSS{0.};
  double covPhiS{0.};
};

/// @brief Residual of a test hit against the pattern line, its uncertainty
///        and its degrees of freedom, PatternStateAux::LineTestRes
struct LineResidual {
  double residual{0.};
  double sigma{0.};
  std::uint8_t nDof{0u};
};

/// @brief PatternStateAux::computeLineResidual
/// @param line The pattern line
/// @param plane The bending plane of the pattern
/// @param nPhiLayers The number of phi layers of the pattern
/// @param anchor The line anchor hit
/// @param last The last inserted hit
/// @param test The hit tested against the line
/// @param beamspot The beam spot
__host__ __device__ inline LineResidual lineResidual(
    const Line& line, const BendPlane& plane, std::uint32_t nPhiLayers,
    const Hit& anchor, const Hit& last, const Hit& test,
    const Beamspot& beamspot) {
  LineResidual res{};

  const bool projectTestHit{(test.flags & CudaHitFlags::measuresPhi) == 0u ||
                            nPhiLayers == 0u};
  const Vector3 testPos{
      projectTestHit ? projectToPhiPlane(test, plane.normal, plane.offset)
                     : test.position};
  const Vector3 K{testPos - line.position};
  const double KdotD{dot(K, line.direction)};

  const Vector3 R{K - KdotD * line.direction};
  res.residual = norm(R);
  if (res.residual < epsilon) {
    res.residual = DBL_MAX;
    return res;
  }
  const Vector3 resDir{R / res.residual};
  const double alpha{KdotD / line.leverArm};

  double phiPlaneDerivativeAcc{0.};
  double phiDisplDerivativeAcc{0.};
  double residualCovAcc{0.};

  auto covarianceTerm = [&](const Hit& hit, const Vector3& pos,
                            const double preFactor, bool isProjected) {
    if (!isProjected) {
      residualCovAcc +=
          square(preFactor) * intrinsicVariance(hit, resDir, false);
      return;
    }
    const Vector3& sensorDir{hit.sensorDirection};
    const double projFactor{dot(sensorDir, resDir) /
                            dot(sensorDir, plane.normal)};
    const Vector3 trfDir{resDir - projFactor * plane.normal};

    residualCovAcc += square(preFactor) * intrinsicVariance(hit, trfDir, true);
    phiPlaneDerivativeAcc += preFactor * projFactor * perp(pos);
    phiDisplDerivativeAcc += preFactor * projFactor;
  };

  if (line.useBeamspot) {
    const double covS1 = square(beamspot.length) * square(resDir.z) +
                         square(beamspot.radius) * (1 - square(resDir.z));
    residualCovAcc += square(alpha - 1) * covS1;
  } else {
    covarianceTerm(anchor, line.position, alpha - 1, true);
  }

  const Vector3 pos2{line.position + line.leverArm * line.direction};
  covarianceTerm(last, pos2, -alpha, true);

  covarianceTerm(test, testPos, 1., projectTestHit);

  const double projSigmaTerm{square(phiPlaneDerivativeAcc) * plane.covPhiPhi +
                             square(phiDisplDerivativeAcc) * plane.covSS +
                             2 * phiPlaneDerivativeAcc * phiDisplDerivativeAcc *
                                 plane.covPhiS};

  res.sigma = std::sqrt(residualCovAcc + projSigmaTerm);
  res.nDof = projectTestHit ? 1u : 2u;
  return res;
}

/// @brief PatternStateAux::angleToBendPlane: the angle between a point and
///        the bending plane, as seen from the origin
__host__ __device__ inline double angleToBendPlane(const BendPlane& plane,
                                                   const Vector3& pos) {
  const double dist{dot(plane.normal, pos) - plane.offset};
  const double rho{plane.normal.y * pos.x - plane.normal.x * pos.y};
  return std::atan2(dist, rho);
}

/// @brief PatternStateAux::phiAtRadius: the approximate azimuth of the
///        bending plane at a transverse distance from the beam line
__host__ __device__ inline double phiAtRadius(const BendPlane& plane,
                                              double radius) {
  return plane.phi + plane.offset / radius;
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
template <typename ForEachPhiHit>
__host__ __device__ inline BendPlane fitBendPlane(
    std::uint32_t nPhiLayers, double previousPhi, int sector,
    const Beamspot& beamspot, ForEachPhiHit&& forEachPhiHit) {
  BendPlane plane{};
  // The solution closest to the sector's direction
  auto alignPhiAndStore = [&](double newPatPhi) {
    const double refPhi{ExpandedSector::phi(sector)};
    while (newPatPhi - refPhi > pi / 2.) {
      newPatPhi -= pi;
    }
    while (newPatPhi - refPhi < -pi / 2.) {
      newPatPhi += pi;
    }
    plane.phi = newPatPhi;
    plane.normal = Vector3{-std::sin(plane.phi), std::cos(plane.phi), 0.};
  };
  if (nPhiLayers == 0u) {
    alignPhiAndStore(ExpandedSector::phi(sector));
    plane.offset = 0.;
    plane.covPhiPhi = square(ExpandedSector::size(sector)) / 3.;
    plane.covSS = 0.;
    plane.covPhiS = 0.;
    return plane;
  }
  if (nPhiLayers == 1u) {
    // A copy: the visitor may hand over a hit that lives only for the call
    Hit phiHit{};
    forEachPhiHit([&](const Hit& hit) { phiHit = hit; });
    const double bsX{beamspot.position.x};
    const double bsY{beamspot.position.y};
    const double dX{phiHit.position.x - bsX};
    const double dY{phiHit.position.y - bsY};
    alignPhiAndStore(std::atan2(dY, dX));
    plane.offset = dot(plane.normal, beamspot.position);

    const double eRX{plane.normal.y};
    const double eRY{-plane.normal.x};
    const double lever{dX * eRX + dY * eRY};
    const double rhoBS{bsX * eRX + bsY * eRY};
    const double rhoHit{rhoBS + lever};

    const double varBS{square(beamspot.radius)};
    const double varHit{intrinsicVariance(phiHit, plane.normal, false)};
    const double invL2{1. / square(lever)};

    plane.covPhiPhi = (varHit + varBS) * invL2;
    plane.covPhiS = -(rhoBS * varHit + rhoHit * varBS) * invL2;
    plane.covSS = (square(rhoBS) * varHit + square(rhoHit) * varBS) * invL2;
    return plane;
  }

  const Vector3 n0{-std::sin(previousPhi), std::cos(previousPhi), 0.};
  const double bsX{beamspot.position.x};
  const double bsY{beamspot.position.y};
  double weightSum{0.};
  double Sxx{0.};
  double Sxy{0.};
  double Syy{0.};
  double centroidX{0.};
  double centroidY{0.};
  forEachPhiHit([&](const Hit& hit) {
    const double variance{intrinsicVariance(hit, n0, false)};
    if (variance < epsilon) {
      return;
    }
    const double weight{1. / variance};
    const double dX{hit.position.x - bsX};
    const double dY{hit.position.y - bsY};
    const double wdX{weight * dX};
    const double wdY{weight * dY};

    weightSum += weight;
    centroidX += wdX;
    centroidY += wdY;
    Sxx += wdX * dX;
    Sxy += wdX * dY;
    Syy += wdY * dY;
  });
  weightSum += 1. / square(beamspot.radius);

  const double weightSumInv{1. / weightSum};
  centroidX *= weightSumInv;
  centroidY *= weightSumInv;

  const double Mxx{Sxx - weightSum * square(centroidX)};
  const double Mxy{Sxy - weightSum * centroidX * centroidY};
  const double Myy{Syy - weightSum * square(centroidY)};

  const double eigenvalSeparation{
      std::sqrt(square(Mxx - Myy) + square(2. * Mxy))};
  const double eigenValSeparationInv{1. / eigenvalSeparation};

  alignPhiAndStore(0.5 * std::atan2(2. * Mxy, Mxx - Myy));

  centroidX += bsX;
  centroidY += bsY;
  plane.offset = plane.normal.x * centroidX + plane.normal.y * centroidY;
  const double eRX{plane.normal.y};
  const double eRY{-plane.normal.x};
  const double rhoCentroid{centroidX * eRX + centroidY * eRY};

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

/// @brief Sizes of the device buffers of the pattern finding. The finding
///        runs once per event, with buffers of fixed size; what does not fit
///        is dropped and reported, never built again.
struct CudaPatternFinderSizes {
  /// @brief Seeds built at once before their patterns are accepted
  std::size_t batchSize{1024u};
  /// @brief Device memory for the patterns being built, shared by the seeds
  ///        of a batch. A seed that needs more room than its share has no
  ///        patterns and is reported
  std::size_t workspaceBytes{std::size_t{1} << 30u};
  /// @brief Hits of one pattern, at most one per detector layer it crosses
  std::size_t maxHitsPerPattern{64u};
  /// @brief Accepted patterns of the event
  std::size_t maxPatterns{4096u};
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
  /// @brief Device: copy of the event's patterns to the host
  double downloadUs{0.};
  /// @brief Host: phi-only hits and the conversion of the patterns
  double tailUs{0.};
  std::size_t nSeeds{0u};
  std::size_t nCandidates{0u};
  /// @brief Patterns in eta accepted on the device
  std::size_t nPatternsInEta{0u};
  /// @brief Patterns each seed had room for, from the workspace budget
  std::size_t capacityPerSeed{0u};
  /// @brief CudaPatternOverflow bits of what did not fit, whose seeds have no
  ///        patterns; zero if everything fitted
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
/// more hits in a pattern than they have hit slots, writes no pattern; its
/// overflow column holds the CudaPatternOverflow bits of what did not fit. The
/// call waits only for the stream.
void buildCudaPatterns(const CudaHitPayloadContainer& hits,
                       const CudaCandidateListContainer& lists,
                       const CudaPatternFinderConfig& cfg,
                       std::size_t firstSeed,
                       CudaPatternRecordContainer& patterns,
                       cudaStream_t stream);

/// @brief Finds the patterns in eta of an event on the device, in one pass.
///
/// The seeds are prepared, then built in batches in tree order, each seed in
/// the room the workspace budget gives it. After each batch, one thread
/// accepts its patterns seed by seed: a seed whose hit is already in
/// maxSeedAttempts accepted patterns is skipped, and each new pattern either
/// replaces the accepted patterns it overlaps or loses to one of them. A seed
/// that did not fit its room, and a pattern that did not fit the event's
/// slots, are left out; the overflow bits of entry 0 say what did not fit.
/// The hits and the lists must be on the device. Returns the event's patterns
/// on the host as the slots of entry 0: the accepted ones have status ok, the
/// dropped ones status rejected. The call synchronizes the stream.
CudaPatternRecordContainer findCudaPatternsInEta(
    const CudaHitPayloadContainer& hits, CudaCandidateListContainer& lists,
    const CudaPatternFinderConfig& cfg, const CudaPatternFinderSizes& sizes,
    cudaStream_t stream, CudaPatternFinderTiming* timing = nullptr);

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

/// @brief Finds the patterns in eta of an event on the device, version 2, in
///        one pass.
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
/// 0. sizes.batchSize and sizes.maxHitsPerPattern are not used.
CudaPatternRecordContainer findCudaPatternsInEtaV2(
    const CudaHitPayloadContainer& hits, CudaCandidateListContainer& lists,
    const CudaPatternFinderConfig& cfg, const CudaPatternFinderSizes& sizes,
    cudaStream_t stream, CudaPatternFinderTiming* timing = nullptr);

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
    /// @brief 1: findCudaPatternsInEta, 2: findCudaPatternsInEtaV2
    unsigned version{1u};
    /// @brief Version 2: the tracking geometry, whose surfaces go to the device
    ///        once, and whether MDT hits take part
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry{};
    bool useMdtHits{true};
    /// @brief Version 2: the share of the device's free memory the pattern
    ///        building takes, measured at the first event. Zero takes
    ///        sizes.workspaceBytes instead
    double workspaceFraction{0.5};
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

  /// @brief Finds the patterns of an event with version 2, converted to the
  ///        output of the algorithm. The host computes
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
  MuonGlobalPatternContainer findPatternsV2(
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
  ///        of version 2
  struct SurfaceTable {
    std::uint64_t* geometryId{nullptr};
    double* rotation{nullptr};
    std::uint32_t nSurfaces{0u};
  };
  const SurfaceTable& surfaceTable(const Acts::GeometryContext& gctx) const;
  mutable SurfaceTable m_surfaces{};
  mutable std::once_flag m_surfacesOnce{};
  /// @brief The bytes of the pattern building of version 2, from
  ///        workspaceFraction; zero takes sizes.workspaceBytes
  mutable std::size_t m_arenaBytes{0u};

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
///        search tree and the conversion of the patterns are those of the CPU
///        algorithm.
class CudaGlobalPatternFinderAlgorithm final : public IAlgorithm {
 public:
  struct Config : GlobalPatternFinderAlgorithm::Config {
    /// @brief CudaPatternFinderSizes
    std::size_t batchSize{1024u};
    std::size_t workspaceBytes{std::size_t{1} << 30u};
    std::size_t maxHitsPerPattern{64u};
    std::size_t maxPatterns{4096u};
    /// @brief Version of the device pattern finding, 1 or 2, see
    ///        CudaGlobalPatternFinder::Config
    unsigned version{1u};
    /// @brief Version 2: share of the device's free memory for the pattern
    ///        building, see CudaGlobalPatternFinder::Config
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
  /// @brief Version 1: the CPU search tree and payloads, then the device
  MuonGlobalPatternContainer findPatternsV1(
      const Acts::GeometryContext& gctx,
      const MuonSpacePointContainer& inSpacePoints,
      const GlobalPatternFinder_t::BeamspotInfo& beamSpot,
      CudaPatternFinderTiming& timing) const;

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
