// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFinding/GlobalPatternFinderDefs.hpp"

#include "Acts/Definitions/Tolerance.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/Utilities/Enumerate.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "Acts/Utilities/MathHelpers.hpp"
#include "Acts/Utilities/UnitVectors.hpp"
#include "Acts/Utilities/VectorHelpers.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <format>
#include <numbers>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace Acts::UnitLiterals;

namespace {

/// @brief Parts of Muon::MuonSectorMapping (MuonDetDescrUtils) used by the ExpandedSector.
///        Free functions instead of a class; called as sectorMap::f() in place
///        of sectorMap.f()
namespace sectorMap {

constexpr double s_oneEightsOfPi{std::numbers::pi / 8.};         //  pi/8
constexpr double s_inverseOneEightsOfPi{8. / std::numbers::pi};  // 8/pi
constexpr std::array<double, 2> s_sectorSize{
    0.4 * s_oneEightsOfPi,
    0.6 * s_oneEightsOfPi};  // side of a sector in radiants
constexpr double s_sectorOverlap{
    0.1 *
    s_oneEightsOfPi};  // size of the overlap between small and large sectors

/** sector size (exclusive) in radians */
double sectorSize(int sector) {
  const int idx = sector % 2;
  return s_sectorSize[idx];
}

/** sector width (with overlap) in radians */
double sectorWidth(int sector) {
  return sectorSize(sector) + s_sectorOverlap;
}

/** returns the centeral phi position of a sector in radians */
double sectorPhi(int sector) {
  if (sector < 10)
    return std::numbers::pi * (sector - 1) / 8.;
  return -std::numbers::pi * (2 - (sector - 1) / 8.);
}

/** transforms a phi position from and to the sector coordinate system in
 * radians */
double transformPhiToSector(double phi, int sector, bool toSector = true) {
  double sign = toSector ? -1 : 1;
  double dphi = phi + sign * sectorPhi(sector);
  if (dphi > std::numbers::pi)
    dphi -= 2 * std::numbers::pi;
  if (dphi < -std::numbers::pi)
    dphi += 2 * std::numbers::pi;
  return dphi;
}

/** checks whether the phi position is consistent with sector */
bool insideSector(int sector, double phi) {
  double phiInSec = transformPhiToSector(phi, sector);
  if (phiInSec < -sectorWidth(sector))
    return false;
  if (phiInSec > sectorWidth(sector))
    return false;
  return true;
}

/** returns the sector corresponding to the phi position */
int getSector(double phi) {
  // remap phi onto sector structure
  double val = (phi + sectorSize(1)) *
               s_inverseOneEightsOfPi;  // convert to value between -8 and 8,
                                        // shift by width first sector
  if (val < 0)
    val += 16;
  int sliceIndex = static_cast<int>(val / 2);  // large/small wedge
  double valueInSlice = val - 2 * sliceIndex;
  int sector = 2 * sliceIndex + 1;
  if (valueInSlice > 1.2)
    ++sector;
  return sector;
}

/** returns the main sector plus neighboring if the phi position is in an
 * overlap region */
void getSectors(double phi, std::vector<int>& sectors) {
  int sector = getSector(phi);
  int sectorNext = sector != 16 ? sector + 1 : 1;
  int sectorBefore = sector != 1 ? sector - 1 : 16;
  if (insideSector(sectorBefore, phi))
    sectors.push_back(sectorBefore);
  if (insideSector(sector, phi))
    sectors.push_back(sector);
  if (insideSector(sectorNext, phi))
    sectors.push_back(sectorNext);
}

/** returns the phi position of the overlap between the two sectors (which have
 * to be neighboring) in radians */
double sectorOverlapPhi(int sector1, int sector2) {
  if (sector1 == sector2)
    return sectorPhi(sector1);

  int s1 = std::min(sector1, sector2);
  int s2 = std::max(sector1, sector2);
  if (s2 == 16 && s1 == 1) {
    s1 = 16;
    s2 = 1;
  } else if (std::abs(s1 - s2) > 1) {
    return 0;
  }

  double phi1 = sectorPhi(s1);
  double phio1 = phi1 + sectorSize(s1);
  if (phio1 > std::numbers::pi)
    phio1 -= 2 * std::numbers::pi;

  return phio1;
}

}  // namespace sectorMap

/** @brief Number of sectors in the muon spectrometer (MuonStationIndex) */
constexpr unsigned numberOfSectors() {
  return 16;
}

using SectorProjector = ActsExamples::ExpandedSector::SectorProjector;
constexpr std::int8_t nExpanded = 2 * numberOfSectors();

inline std::tuple<unsigned, SectorProjector> msSectorAndProj(
    const std::int8_t expandSector) {
  assert(expandSector >= 0 && expandSector <= nExpanded);
  if (expandSector == 0) {
    return std::make_tuple(numberOfSectors(), SectorProjector::center);
  } else if (expandSector == 1) {
    return std::make_tuple(numberOfSectors(), SectorProjector::rightOverlap);
  }
  const int regSector = expandSector / 2;
  const int backConv = expandSector - 2 * regSector;
  switch (backConv) {
    case 1:
      return std::make_tuple(regSector, SectorProjector::rightOverlap);
    case 0:
      return std::make_tuple(regSector, SectorProjector::center);
    case -1:
      return std::make_tuple(regSector, SectorProjector::leftOverlap);
    default:
      throw std::runtime_error(
          std::format("Cannot deduce the sector overlap {}", expandSector));
  }
  // all cases have been addressed
}

/** @brief Helper function to construct the gradient of the azimuthal coordinate
 *  @param pos The position vector where the gradient is to be computed
 *  @return The gradient of the azimuthal coordinate */
Acts::Vector3 phiGradient(const Acts::Vector3& pos) {
  return Acts::Vector3{-pos.y(), pos.x(), 0.} /
         Acts::square(Acts::VectorHelpers::perp(pos));
}

/** @brief Column indices of the surface axes, named as in Athena (Amg::x, Amg::y, Amg::z) */
namespace Amg {
constexpr int x{0};
constexpr int y{1};
constexpr int z{2};
}  // namespace Amg

/** @brief Indices of the space point covariance as written by the Athena exporter
 *         (MuonActsDump/SpacePointWriter: covLoc0 = etaCov, covLoc1 = phiCov,
 * covT = timeCov) */
enum class CovIdx : std::uint8_t { etaCov = 0, phiCov = 1, timeCov = 2 };

/** @brief Axes of the measurement surface in the global frame. Athena takes them from
 *         xAOD::muonSurface(sp->primaryMeasurement()). The example EDM has no
 * surface, so they are rebuilt from the space point, which Athena defines from
 * that surface: y = strip direction (sensorDirection), z = plane normal, x = y
 * cross z (measuring direction). Signs may differ from the surface axes; every
 * use is squared or a line direction. */
Acts::RotationMatrix3 measurementAxes(const ActsExamples::MuonSpacePoint& sp,
                                      const Acts::RotationMatrix3& toGlobal) {
  const Acts::Vector3 y{sp.sensorDirection().normalized()};
  const Acts::Vector3 z{sp.planeNormal()};
  Acts::RotationMatrix3 axes{};
  axes.col(Amg::x) = toGlobal * y.cross(z);
  axes.col(Amg::y) = toGlobal * y;
  axes.col(Amg::z) = toGlobal * z;
  return axes;
}

}  // namespace

namespace ActsExamples {

/** @brief Transformation from the bucket (sector) frame into the global frame.
 *         Replaces Athena's bucket->msSector()->localToGlobalTransform(gctx).
 *  TODO: return the transform carried by the bucket once MuonSpacePointBucket
 * provides it. */
Acts::Transform3 localToGlobalTransform(
    const Acts::GeometryContext& /*gctx*/,
    const MuonSpacePointBucket& /*bucket*/) {
  throw std::runtime_error(
      "GlobalPatternFinderAlgorithm: the MuonSpacePointBucket does not carry "
      "its local to global transform yet");
}

ExpandedSector::ExpandedSector(const std::int8_t expSector)
    : m_sector{expSector} {}
ExpandedSector::ExpandedSector(const unsigned msSector,
                               const SectorProjector proj) {
  m_sector = (2 * msSector + Acts::toUnderlying(proj)) % nExpanded;
}

ExpandedSector::ExpandedSector(const double phi) {
  std::vector<int> sectors{};
  sectorMap::getSectors(phi, sectors);
  assert(!sectors.empty());
  if (sectors.size() == 1) {
    (*this) = ExpandedSector{static_cast<unsigned>(sectors[0]),
                             SectorProjector::center};
  } else {
    const int dS = (sectors[1] - sectors[0]) % numberOfSectors();
    assert(std::abs(dS) == 1);
    (*this) = ExpandedSector{static_cast<unsigned>(sectors[0]),
                             static_cast<SectorProjector>(dS)};
  }
}

SectorProjector ExpandedSector::projector() const {
  return std::get<1>(msSectorAndProj(sector()));
}
unsigned ExpandedSector::msSector() const {
  return std::get<0>(msSectorAndProj(sector()));
}
std::int8_t ExpandedSector::sector() const {
  return m_sector;
}
unsigned ExpandedSector::adjacentMsSector() const {
  const auto [msSec, proj] = msSectorAndProj(sector());
  if (msSec == 1 && proj == SectorProjector::leftOverlap) {
    return numberOfSectors();
  } else if (msSec == numberOfSectors() &&
             proj == SectorProjector::rightOverlap) {
    return 1;
  }
  return msSec + Acts::toUnderlying(proj);
}
double ExpandedSector::sectorSize() const {
  unsigned sector1{msSector()};
  unsigned sector2{adjacentMsSector()};

  if (sector1 == sector2) {
    return sectorMap::sectorSize(sector1);
  }
  /** The overlap size is the same for small and large sectors */
  return sectorMap::sectorWidth(sector1) - sectorMap::sectorSize(sector1);
}
bool ExpandedSector::operator<(const ExpandedSector& other) const {
  return sector() < other.sector();
}
bool ExpandedSector::operator==(const ExpandedSector& other) const {
  return sector() == other.sector();
}
bool ExpandedSector::operator!=(const ExpandedSector& other) const {
  return sector() != other.sector();
}
double ExpandedSector::phi() const {
  return sectorMap::sectorOverlapPhi(msSector(), adjacentMsSector());
}
Acts::Vector3 ExpandedSector::radialDir() const {
  return Acts::makeDirectionFromPhiTheta(phi(), 90._degree);
}
Acts::Vector3 ExpandedSector::normalDir() const {
  return Acts::makeDirectionFromPhiTheta(phi() + 90._degree, 90._degree);
}
bool ExpandedSector::isNeighbour(const ExpandedSector& other) const {
  const int dS = (other.sector() - sector()) % nExpanded;
  return std::abs(dS) <= 1;
}
bool ExpandedSector::insideSector(const double phi) const {
  if (sectorMap::insideSector(msSector(), phi) &&
      sectorMap::insideSector(adjacentMsSector(), phi)) {
    return true;
  }
  return false;
}

std::string ExpandedSector::toString(const SectorProjector proj) {
  switch (proj) {
    using enum SectorProjector;
    case leftOverlap:
      return "leftOverlap";
    case center:
      return "center";
    case rightOverlap:
      return "rightOverlap";
  }
  return "";
}
std::ostream& ExpandedSector::toString(std::ostream& ostr) const {
  ostr << "Expanded sector: " << static_cast<int>(sector()) << " ->  "
       << projector() << " of sector " << msSector();
  return ostr;
}

StIndex toStationIndex(MuonSpacePoint::MuonId::StationName index) {
  using ChIndex = MuonSpacePoint::MuonId::StationName;
  switch (index) {
    case ChIndex::BIS:
    case ChIndex::BIL:
      return StIndex::BI;
    case ChIndex::BMS:
    case ChIndex::BML:
      return StIndex::BM;
    case ChIndex::BOL:
    case ChIndex::BOS:
      return StIndex::BO;
    case ChIndex::BEE:
      return StIndex::BE;
    case ChIndex::EIL:
    case ChIndex::EIS:
      return StIndex::EI;
    case ChIndex::EML:
    case ChIndex::EMS:
      return StIndex::EM;
    case ChIndex::EOL:
    case ChIndex::EOS:
      return StIndex::EO;
    case ChIndex::EEL:
    case ChIndex::EES:
      return StIndex::EE;
    /// Don't do anything for
    case ChIndex::MaxVal:
    case ChIndex::UnDef:
      break;
  };
  return StIndex::StUnknown;
}
LayerIndex toLayerIndex(StIndex index) {
  switch (index) {
    case StIndex::BI:
    case StIndex::EI:
      return LayerIndex::Inner;
    case StIndex::BM:
    case StIndex::EM:
      return LayerIndex::Middle;
    case StIndex::EO:
    case StIndex::BO:
      return LayerIndex::Outer;
    case StIndex::BE:
      return LayerIndex::BarrelExtended;
    case StIndex::EE:
      return LayerIndex::Extended;
    case StIndex::StUnknown:
    case StIndex::StIndexMax:
      break;
  }
  return LayerIndex::LayerIndexMax;
}
bool isBarrel(const StIndex index) {
  switch (index) {
    case StIndex::BI:
    case StIndex::BM:
    case StIndex::BO:
    case StIndex::BE:
      return true;
    default:
      return false;
  }
}

bool isPrecisionHit(const MuonSpacePoint& hit) {
  using enum MuonSpacePoint::MuonId::TechField;
  /** Athena requires the primary sTgc measurement to be a strip. The channel
   * type is not exported, so only strip-only space points (eta without phi) are
   * taken as precision. */
  return hit.id().technology() == Mdt || hit.id().technology() == Mm ||
         (hit.id().technology() == sTgc && hit.id().measuresEta() &&
          !hit.id().measuresPhi());
}

SeedSelector::SeedSelector(Config&& config) : m_cfg{config} {
  if (m_cfg.seedFromInner) {
    m_seedinglayers.push_back(LayerIndex::Inner);
  }
}

bool SeedSelector::goodForSeeding(const HitPayload& hit) const {
  const LayerIndex hitLayer{toLayerIndex(hit.station)};
  if (std::ranges::find(m_seedinglayers, hitLayer) == m_seedinglayers.end() ||
      (hit.spacePoint()->isStraw() && !m_cfg.seedFromMdt)) {
    return false;
  }
  return true;
}
double SeedSelector::thetaSearchWindow(const HitPayload& hit) const {
  const LayerIndex hitLayer{toLayerIndex(hit.station)};
  return (hitLayer == LayerIndex::Inner || hitLayer == LayerIndex::Outer)
             ? m_cfg.thetaSearchWindow
             : 0.5 * m_cfg.thetaSearchWindow;
}

using MuonId = MuonSpacePoint::MuonId;

HitPayload::HitPayload(const Acts::GeometryContext& gctx,
                       const MuonSpacePoint* sp,
                       const MuonSpacePointBucket* parentBucket,
                       const Acts::Transform3& localToGlobal)
    : position{localToGlobal * sp->localPosition()},
      underlyingSp{sp},
      bucket{parentBucket} {
  if (!sp->id().measuresEta()) {
    // Phi-only measurements
    const Acts::Vector3 phiMeasDir{localToGlobal.rotation() *
                                   sp->toNextSensor()};

    phiCov = sp->covariance()[Acts::toUnderlying(CovIdx::phiCov)] *
             Acts::square(phiMeasDir.dot(phiGradient(position)));
    return;
  }
  const Acts::RotationMatrix3 surfLinearTrf{
      measurementAxes(*sp, localToGlobal.rotation())};

  if (sp->isStraw()) {
    // Remember that for straw hits, the x component of secondaryMeasDir is
    // repurposed to store the transverse covariance of the drift radius
    double& discCov = stripAngle;
    discCov = Acts::square(sp->driftRadius()) +
              sp->covariance()[Acts::toUnderlying(CovIdx::etaCov)];

    if (sp->id().measuresPhi()) {
      phiCov =
          discCov / Acts::square(Acts::VectorHelpers::perp(position)) +
          Acts::square(globalSensorDirection(gctx).dot(phiGradient(position))) *
              (sp->covariance()[Acts::toUnderlying(CovIdx::phiCov)] - discCov);
    }
  } else if (sp->id().measuresPhi()) {
    const Acts::Vector3 phiMeasDir = surfLinearTrf.col(Amg::y);
    const Acts::Vector3 gradPhi{phiGradient(position)};
    /** @brief Helper method to compute the contribution of a 1D measurement to the residual variance */
    auto oneDimContribution = [&](CovIdx idx,
                                  const Acts::Vector3& measDir) -> double {
      return sp->covariance()[Acts::toUnderlying(idx)] *
             Acts::square(measDir.dot(gradPhi));
    };

    // Handle the case of TGC separately
    if (sp->id().technology() == MuonId::TechField::Tgc) {
      const Acts::Vector3 etaMeasDir =
          localToGlobal.rotation() * sp->toNextSensor();
      const Acts::Vector3 phiSensorDir = surfLinearTrf.col(Amg::x);

      const double c{etaMeasDir.dot(phiMeasDir)};
      if (std::abs(c) > Acts::s_epsilon) {
        stripAngle = std::atan2(etaMeasDir.dot(phiSensorDir), c);
        nonOrthogonalStrips = true;
      }
      phiCov = oneDimContribution(CovIdx::etaCov, etaMeasDir) +
               oneDimContribution(CovIdx::phiCov, phiMeasDir);
    } else {
      const Acts::Vector3 etaMeasDir = surfLinearTrf.col(Amg::x);
      phiCov = oneDimContribution(CovIdx::etaCov, etaMeasDir) +
               oneDimContribution(CovIdx::phiCov, phiMeasDir);
    }
  }
}
const Acts::Vector3& HitPayload::globalPosition(
    const Acts::GeometryContext& /*gctx*/) const {
  return position;
}
Acts::Vector3 HitPayload::globalSensorDirection(
    const Acts::GeometryContext& gctx) const {
  const Acts::RotationMatrix3 toGlobal{
      localToGlobalTransform(gctx, *bucket).linear()};

  if (spacePoint()->isStraw()) {
    return toGlobal * spacePoint()->sensorDirection();
  } else {
    const Acts::RotationMatrix3 surfLinearTrf{
        measurementAxes(*spacePoint(), toGlobal)};
    if (nonOrthogonalStrips) {
      return -std::sin(stripAngle) * surfLinearTrf.col(Amg::y) +
             std::cos(stripAngle) * surfLinearTrf.col(Amg::x);
    }
    return surfLinearTrf.col(Amg::y);
  }
}
double HitPayload::intrinsicVariance(
    const Acts::GeometryContext& gctx,
    const Acts::Vector3& contractionVector) const {
  if (spacePoint()->isStraw()) {
    const double discCov{stripAngle};
    const double dotProdSq{
        Acts::square(globalSensorDirection(gctx).dot(contractionVector))};
    const double etaTerm{discCov *
                         (contractionVector.squaredNorm() - dotProdSq)};
    if (spacePoint()->id().measuresPhi()) {
      return etaTerm +
             dotProdSq *
                 spacePoint()->covariance()[Acts::toUnderlying(CovIdx::phiCov)];
    } else {
      return etaTerm;
    }
  } else if (spacePoint()->id().measuresEta()) {
    const Acts::RotationMatrix3 surfLinearTrf{measurementAxes(
        *spacePoint(), localToGlobalTransform(gctx, *bucket).linear())};
    /** @brief Helper method to compute the contribution of a 1D measurement to the residual variance */
    auto oneDimContribution = [&](CovIdx idx,
                                  const Acts::Vector3& measDir) -> double {
      return spacePoint()->covariance()[Acts::toUnderlying(idx)] *
             Acts::square(measDir.dot(contractionVector));
    };
    if (spacePoint()->id().measuresPhi()) {
      const Acts::Vector3 etaMeasDir{
          nonOrthogonalStrips ? Acts::Vector3{globalSensorDirection(gctx).cross(
                                    surfLinearTrf.col(Amg::z))}
                              : Acts::Vector3{surfLinearTrf.col(Amg::x)}};
      const Acts::Vector3 phiMeasDir{surfLinearTrf.col(Amg::y)};
      return oneDimContribution(CovIdx::etaCov, etaMeasDir) +
             oneDimContribution(CovIdx::phiCov, phiMeasDir);
    }
    const Acts::Vector3 etaMeasDir{surfLinearTrf.col(Amg::x)};
    return oneDimContribution(CovIdx::etaCov, etaMeasDir);
  } else {
    throw std::runtime_error(
        "Phi only hits are not meant to be used for residual computation.");
  }
}
double HitPayload::phiVariance(const Acts::GeometryContext& /*gctx*/) const {
  return phiCov;
}
bool HitPayload::operator==(const HitPayload& other) const {
  return spacePoint() == other.spacePoint();
}

bool PatternTopology::layerSorter(const HitPayload& hit1,
                                  const HitPayload& hit2) {
  StIndex st1{hit1.station};
  StIndex st2{hit2.station};
  if (st1 == st2) {
    /** Hits in the same spectrometer sector */
    if (hit1.spacePoint()->id().sameStation(hit2.spacePoint()->id())) {
      if (hit1.locLayer == hit2.locLayer) {
        return hit1.spacePoint()->localPosition().y() <
               hit2.spacePoint()->localPosition().y();
      }
      return hit1.locLayer < hit2.locLayer;
    }
    /** Hits in the same station and different sectors. We can have this case
     * for hits in the overlap region of two adjacent sectors. */
    const double delta{isBarrel(st1)
                           ? Acts::VectorHelpers::perp(hit1.position) -
                                 Acts::VectorHelpers::perp(hit2.position)
                           : std::abs(hit1.position.z()) -
                                 std::abs(hit2.position.z())};
    if (std::abs(delta) <= Acts::s_epsilon) {
      return hit1.spacePoint()->localPosition().y() <
             hit2.spacePoint()->localPosition().y();
    }
    return delta < 0.;
  }
  LayerIndex layer1{toLayerIndex(st1)};
  LayerIndex layer2{toLayerIndex(st2)};
  using enum LayerIndex;
  if (layer1 == layer2) {
    /** Hit in different stations but same station layer. Expected to happen
     * only for Inner and Middle*/
    if (layer1 == Middle) {
      /** If both hits are in the middle layer, the one in the barrel comes
       * first */
      return st1 == StIndex::BM;
    }
    if (layer1 == Inner) {
      /** If both hits are in the inner layer, we use the global R, since in
       * large sector BI comes first, while in small sector EI comes first. */
      return Acts::VectorHelpers::perp(hit1.position) <
             Acts::VectorHelpers::perp(hit2.position);
    }
    throw std::runtime_error(
        "Unexpected to have two pattern-compatible hits one in BO and the "
        "other in EO.");
  }
  if (layer1 == Inner || layer2 == Inner) {
    /** If we have one hit in Inner layer for sure it comes first */
    return layer1 == Inner;
  }
  if (layer1 == Outer || layer2 == Outer) {
    /** If we have one hit in Outer layer for sure it comes last */
    return layer2 == Outer;
  }
  if (layer1 == BarrelExtended || layer2 == BarrelExtended) {
    /** If we have one hit in BarrelExtended and the other in the Middle layer,
     * the former comes first */
    return layer1 == BarrelExtended;
  }
  /** If we have one hit in Extended (EE) layer and the other in the Middle
   * layer, it depends if the latter is endcap or barrel */
  if (layer1 == Extended) {
    return st2 == StIndex::EM;
  }
  return st1 == StIndex::BM;
}

bool PatternTopology::sameLayer(const HitPayload& hit1,
                                const HitPayload& hit2) {
  if (hit1.station != hit2.station) {
    return false;
  }
  if (hit1.locLayer == hit2.locLayer &&
      hit1.spacePoint()->id().sameStation(hit2.spacePoint()->id())) {
    return true;
  }
  const double delta{isBarrel(hit1.station)
                         ? Acts::VectorHelpers::perp(hit1.position) -
                               Acts::VectorHelpers::perp(hit2.position)
                         : std::abs(hit1.position.z()) -
                               std::abs(hit2.position.z())};
  return std::abs(delta) <= Acts::s_epsilon;
}
PatternTopology::GroupIdx PatternTopology::groupIndex(const HitPayload& hit) {
  return static_cast<GroupIdx>(Acts::toUnderlying(hit.station));
}

OnlyPhiHitsProvider::PhiHitsPerGroup OnlyPhiHitsProvider::getPhiOnlyHits(
    const PatternState& pattern, const Acts::GeometryContext& gctx) {
  PhiHitsPerGroup phiOnlyHits{};
  for (const auto& [group, hits] : Acts::enumerate(pattern.hitsPerGroup)) {
    std::vector<const MuonSpacePointBucket*> parentBuckets{};
    for (const auto& hit : hits) {
      const MuonSpacePointBucket* bucket{hit->bucket};
      if (std::ranges::find(parentBuckets, bucket) == parentBuckets.end()) {
        parentBuckets.push_back(bucket);
      }
    }
    for (const MuonSpacePointBucket* bucket : parentBuckets) {
      const Acts::Transform3 localToGlobal{
          localToGlobalTransform(gctx, *bucket)};
      for (const MuonSpacePoint& h : *bucket) {
        if (!h.id().measuresEta()) {
          phiOnlyHits[group].emplace_back(gctx, &h, bucket, localToGlobal);
        }
      }
    }
  }
  return phiOnlyHits;
}

}  // namespace ActsExamples
