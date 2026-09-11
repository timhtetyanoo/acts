// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/EventData/MuonGlobalPatternFinderEvent.hpp"

#include "Acts/Definitions/Units.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "Acts/Utilities/UnitVectors.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <format>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace Acts::UnitLiterals;

namespace ActsExamples {

MuonStationIndex stationIndex(MuonSpacePoint::MuonId::StationName stName) {
  using StationName = MuonSpacePoint::MuonId::StationName;
  switch (stName) {
    case StationName::BIS:
    case StationName::BIL:
      return MuonStationIndex::BI;
    case StationName::BMS:
    case StationName::BML:
      return MuonStationIndex::BM;
    case StationName::BOS:
    case StationName::BOL:
      return MuonStationIndex::BO;
    case StationName::BEE:
      return MuonStationIndex::BE;
    case StationName::EIS:
    case StationName::EIL:
      return MuonStationIndex::EI;
    case StationName::EMS:
    case StationName::EML:
      return MuonStationIndex::EM;
    case StationName::EOS:
    case StationName::EOL:
      return MuonStationIndex::EO;
    case StationName::EES:
    case StationName::EEL:
      return MuonStationIndex::EE;
    default:
      return MuonStationIndex::UnDef;
  }
}

MuonLayerIndex layerIndex(MuonStationIndex stIdx) {
  switch (stIdx) {
    case MuonStationIndex::BI:
    case MuonStationIndex::EI:
      return MuonLayerIndex::Inner;
    case MuonStationIndex::BM:
    case MuonStationIndex::EM:
      return MuonLayerIndex::Middle;
    case MuonStationIndex::BO:
    case MuonStationIndex::EO:
      return MuonLayerIndex::Outer;
    case MuonStationIndex::EE:
      return MuonLayerIndex::Extended;
    case MuonStationIndex::BE:
      return MuonLayerIndex::BarrelExtended;
    default:
      return MuonLayerIndex::UnDef;
  }
}

bool isBarrel(MuonStationIndex stIdx) {
  switch (stIdx) {
    case MuonStationIndex::BI:
    case MuonStationIndex::BM:
    case MuonStationIndex::BO:
    case MuonStationIndex::BE:
      return true;
    default:
      return false;
  }
}

std::string toString(MuonStationIndex stIdx) {
  switch (stIdx) {
    case MuonStationIndex::BI:
      return "BI";
    case MuonStationIndex::BM:
      return "BM";
    case MuonStationIndex::BO:
      return "BO";
    case MuonStationIndex::BE:
      return "BE";
    case MuonStationIndex::EI:
      return "EI";
    case MuonStationIndex::EM:
      return "EM";
    case MuonStationIndex::EO:
      return "EO";
    case MuonStationIndex::EE:
      return "EE";
    default:
      return "Unknown";
  }
}

std::string toString(MuonLayerIndex layIdx) {
  switch (layIdx) {
    case MuonLayerIndex::Inner:
      return "Inner";
    case MuonLayerIndex::Middle:
      return "Middle";
    case MuonLayerIndex::Outer:
      return "Outer";
    case MuonLayerIndex::Extended:
      return "Extended";
    case MuonLayerIndex::BarrelExtended:
      return "BarrelExtended";
    default:
      return "Unknown";
  }
}

std::ostream& operator<<(std::ostream& ostr, MuonStationIndex stIdx) {
  return ostr << toString(stIdx);
}

std::ostream& operator<<(std::ostream& ostr, MuonLayerIndex layIdx) {
  return ostr << toString(layIdx);
}

namespace MuonSectorMapping {
double sectorSize(int sector) {
  const int idx = sector % 2;
  return s_sectorSize[idx];
}
double sectorWidth(int sector) {
  return sectorSize(sector) + s_sectorOverlap;
}
double sectorPhi(int sector) {
  if (sector < 10)
    return std::numbers::pi * (sector - 1) / 8.;
  return -std::numbers::pi * (2 - (sector - 1) / 8.);
}
double transformPhiToSector(double phi, int sector, bool toSector) {
  double sign = toSector ? -1 : 1;
  double dphi = phi + sign * sectorPhi(sector);
  if (dphi > std::numbers::pi)
    dphi -= 2 * std::numbers::pi;
  if (dphi < -std::numbers::pi)
    dphi += 2 * std::numbers::pi;
  return dphi;
}
bool insideSector(int sector, double phi) {
  double phiInSec = transformPhiToSector(phi, sector);
  if (phiInSec < -sectorWidth(sector))
    return false;
  if (phiInSec > sectorWidth(sector))
    return false;
  return true;
}
int getSector(double phi) {
  // remap phi onto the sector structure
  double val = (phi + sectorSize(1)) *
               s_inverseEighthPi;  // convert to value between -8 and 8, shift
                                   // by width first sector
  if (val < 0)
    val += 16;
  int sliceIndex = static_cast<int>(val / 2);  // large/small wedge
  double valueInSlice = val - 2 * sliceIndex;
  int sector = 2 * sliceIndex + 1;
  if (valueInSlice > 1.2)
    ++sector;
  return sector;
}
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
}  // namespace MuonSectorMapping

namespace {

using SectorProjector = MuonExpandedSector::SectorProjector;

constexpr std::int8_t s_nExpanded =
    2 * static_cast<std::int8_t>(MuonSectorMapping::s_numSectors);

std::tuple<unsigned, SectorProjector> msSectorAndProj(
    std::int8_t expandSector) {
  assert(expandSector >= 0 && expandSector <= s_nExpanded);
  if (expandSector == 0) {
    return {MuonSectorMapping::s_numSectors, SectorProjector::center};
  } else if (expandSector == 1) {
    return {MuonSectorMapping::s_numSectors, SectorProjector::rightOverlap};
  }
  const int regSector = expandSector / 2;
  const int backConv = expandSector - 2 * regSector;
  switch (backConv) {
    using enum SectorProjector;
    case 1:
      return {static_cast<unsigned>(regSector), rightOverlap};
    case 0:
      return {static_cast<unsigned>(regSector), center};
    case -1:
      return {static_cast<unsigned>(regSector), leftOverlap};
    default:
      throw std::invalid_argument(
          "MuonExpandedSector: cannot deduce the sector overlap");
  }
  // all cases have been addressed
}

}  // namespace

MuonExpandedSector::MuonExpandedSector(unsigned msSector,
                                       SectorProjector proj) {
  if (msSector < 1 || msSector > MuonSectorMapping::s_numSectors) {
    throw std::invalid_argument("MuonExpandedSector: msSector is out of range");
  }
  m_sector = static_cast<std::int8_t>(
      (2 * msSector + Acts::toUnderlying(proj)) % s_nExpanded);
}

MuonExpandedSector::MuonExpandedSector(double phi) {
  std::vector<int> sectors{};
  MuonSectorMapping::getSectors(phi, sectors);
  if (sectors.empty()) {
    throw std::invalid_argument(
        "MuonExpandedSector: no MS sector for the given phi");
  }
  if (sectors.size() == 1) {
    *this = MuonExpandedSector{static_cast<unsigned>(sectors[0]),
                               SectorProjector::center};
  } else {
    const int dS = static_cast<int>((sectors[1] - sectors[0]) %
                                    MuonSectorMapping::s_numSectors);
    if (std::abs(dS) != 1) {
      throw std::invalid_argument(
          "MuonExpandedSector: overlap sectors are not neighbours");
    }
    *this = MuonExpandedSector{static_cast<unsigned>(sectors[0]),
                               static_cast<SectorProjector>(dS)};
  }
}

MuonExpandedSector::MuonExpandedSector(std::int8_t expandedSector)
    : m_sector{expandedSector} {
  if (expandedSector < 0 || expandedSector >= s_nExpanded) {
    throw std::invalid_argument(
        "MuonExpandedSector: expanded sector is out of range");
  }
}

bool MuonExpandedSector::operator<(const MuonExpandedSector& other) const {
  return sector() < other.sector();
}

bool MuonExpandedSector::operator==(const MuonExpandedSector& other) const {
  return sector() == other.sector();
}

bool MuonExpandedSector::operator!=(const MuonExpandedSector& other) const {
  return sector() != other.sector();
}

unsigned MuonExpandedSector::msSector() const {
  return std::get<0>(msSectorAndProj(sector()));
}

unsigned MuonExpandedSector::adjacentMsSector() const {
  const auto [msSec, proj] = msSectorAndProj(sector());
  if (msSec == 1 && proj == SectorProjector::leftOverlap) {
    return MuonSectorMapping::s_numSectors;
  } else if (msSec == MuonSectorMapping::s_numSectors &&
             proj == SectorProjector::rightOverlap) {
    return 1;
  }
  return msSec + Acts::toUnderlying(proj);
}

MuonExpandedSector::SectorProjector MuonExpandedSector::projector() const {
  return std::get<1>(msSectorAndProj(sector()));
}

std::int8_t MuonExpandedSector::sector() const {
  return m_sector;
}

double MuonExpandedSector::phi() const {
  return MuonSectorMapping::sectorOverlapPhi(msSector(), adjacentMsSector());
}

Acts::Vector3 MuonExpandedSector::radialDir() const {
  return Acts::makeDirectionFromPhiTheta(phi(), 90_degree);
}

Acts::Vector3 MuonExpandedSector::normalDir() const {
  return Acts::makeDirectionFromPhiTheta(phi() + 90_degree, 90_degree);
}

bool MuonExpandedSector::isNeighbour(const MuonExpandedSector& other) const {
  const int dS = (other.sector() - sector()) % s_nExpanded;
  return std::abs(dS) <= 1;
}

std::string MuonExpandedSector::toString(SectorProjector projector) {
  switch (projector) {
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

std::ostream& MuonExpandedSector::print(std::ostream& ostr) const {
  ostr << "Expanded sector: " << static_cast<int>(sector()) << " -> "
       << projector() << " of sector " << msSector();
  return ostr;
}

MuonGlobalPattern::MuonGlobalPattern(HitCollection&& hitPerStation,
                                     BucketCollection&& bucketPerStation)
    : m_hitsInStation(std::move(hitPerStation)),
      m_parentBuckets(std::move(bucketPerStation)) {}

double MuonGlobalPattern::sectorPhi() const {
  return m_sector.phi();
}

std::vector<MuonGlobalPattern::StIndex> MuonGlobalPattern::getStations() const {
  std::vector<StIndex> out{};
  out.reserve(m_hitsInStation.size());
  std::ranges::transform(m_hitsInStation, std::back_inserter(out),
                         [](const auto& pair) { return pair.first; });
  return out;
}

const std::vector<MuonGlobalPattern::HitType>& MuonGlobalPattern::hitsInStation(
    StIndex station) const {
  const auto it = m_hitsInStation.find(station);
  if (it != m_hitsInStation.end()) {
    return it->second;
  }
  static const std::vector<HitType> empty{};
  return empty;
}

const std::vector<const MuonSpacePointBucket*>&
MuonGlobalPattern::bucketsInStation(StIndex station) const {
  const auto it = m_parentBuckets.find(station);
  if (it != m_parentBuckets.end()) {
    return it->second;
  }
  static const std::vector<const MuonSpacePointBucket*> empty{};
  return empty;
}

std::ostream& MuonGlobalPattern::print(std::ostream& ostr) const {
  ostr << std::format(
      "SpacePoint Pattern, Sector: {} & {}, theta: {}, Phi: {}, Sector Phi: "
      "{}, nPrecisionLayers: {}, nTriggerLayers: {}, nPhiLayers: {}, mean "
      "normalized residual squared: {}",
      sector(), isSectorOverlap() ? std::to_string(secondarySector()) : "-",
      theta() / 1_degree, phi() / 1_degree, sectorPhi() / 1_degree,
      nPrecisionLayers(), nTriggerLayers(), nPhiLayers(), meanNormResidual2());
  ostr << ", Hit per station: \n";
  for (const auto& [station, hits] : m_hitsInStation) {
    ostr << std::format(" Station {}: {} hits\n", toString(station),
                        hits.size());
    for (const auto& hit : hits) {
      ostr << " " << *hit << "\n";
    }
  }
  return ostr;
}

}  // namespace ActsExamples
