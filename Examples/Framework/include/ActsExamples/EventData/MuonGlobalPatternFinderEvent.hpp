// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Utilities/OstreamFormatter.hpp"
#include "ActsExamples/EventData/MuonSpacePoint.hpp"

#include <array>
#include <cstdint>
#include <numbers>
#include <ostream>
#include <string>
#include <vector>

namespace ActsExamples {

/// @brief Classification of the MS station layers
enum class MuonStationIndex : std::int8_t {
  UnDef = -1,  ///< Undefined station
  BI,          ///< Barrel inner
  BM,          ///< Barrel middle
  BO,          ///< Barrel outer
  BE,          ///< Barrel endcap extension
  EI,          ///< Endcap inner
  EM,          ///< Endcap middle
  EO,          ///< Endcap outer
  EE,          ///< Endcap extension
  MaxVal       ///< Number of stations
};

/// @brief Classification of the layer inside the MS
enum class MuonLayerIndex : std::int8_t {
  UnDef = -1,      ///< Undefined layer
  Inner,           ///< Inner station layer
  Middle,          ///< Middle station layer
  Outer,           ///< Outer station layer
  Extended,        ///< Endcap extension layer
  BarrelExtended,  ///< Barrel endcap-extension layer
  MaxVal           ///< Number of layers
};

/// @brief Maps an MS station name onto its station index
MuonStationIndex stationIndex(MuonSpacePoint::MuonId::StationName stName);
/// @brief Maps a station index onto its layer index
MuonLayerIndex layerIndex(MuonStationIndex stIdx);
/// @brief Returns whether the station index is located in the barrel
bool isBarrel(MuonStationIndex stIdx);
/// @brief Return the station index as a string
std::string toString(MuonStationIndex stIdx);
/// @brief Return the layer index as a string
std::string toString(MuonLayerIndex layIdx);
/// @brief Stream a station index
std::ostream& operator<<(std::ostream& ostr, MuonStationIndex stIdx);
/// @brief Stream a layer index
std::ostream& operator<<(std::ostream& ostr, MuonLayerIndex layIdx);

namespace MuonSectorMapping {
constexpr unsigned int s_numSectors = 16;
constexpr double s_eighthPi = std::numbers::pi / 8.;
constexpr double s_inverseEighthPi = 8. / std::numbers::pi;
constexpr std::array<double, 2> s_sectorSize{0.4 * s_eighthPi,
                                             0.6 * s_eighthPi};
constexpr double s_sectorOverlap = 0.1 * s_eighthPi;

/// @brief Sector size (exclusive) in radians
double sectorSize(int sector);
/// @brief Sector width (with overlap) in radians
double sectorWidth(int sector);
/// @brief Returns the central phi position of a sector in radians
double sectorPhi(int sector);
/// @brief Transforms a phi position from and to the sector coordinate system
///        in radians
double transformPhiToSector(double phi, int sector, bool toSector = true);
/// @brief Checks whether the phi position is consistent with sector
bool insideSector(int sector, double phi);
/// @brief Returns the sector corresponding to the phi position
int getSector(double phi);
/// @brief Returns the main sector plus neighboring if the phi position is in
///        an overlap region
void getSectors(double phi, std::vector<int>& sectors);
/// @brief Returns the phi position of the overlap between the two sectors
///        (which have to be neighboring) in radians
double sectorOverlapPhi(int sector1, int sector2);

}  // namespace MuonSectorMapping

/// @brief Helper functions to describe the expanded sector concept. The expanded
///        sectors are based on the 16 fold symmetry of the MS, but also take
///        into account the overlap between 2 sectors.
///
///        The regular msSector is multiplied by 2 and then the sectorProjector
///        is added which can be either -1 to indicate that the overlap between
///        the current sector and the left sector is of interest, or 0 to
///        indicate that the sector center is of interest and finally 1 to
///        indicate that the sector to the right is of interest.
class MuonExpandedSector {
 public:
  /// @brief Enumeration to select the sector projection of the
  ///        regular MS sector
  enum class SectorProjector : std::int8_t {
    leftOverlap =
        -1,  ///< Project the segment onto the overlap with the previous sector
    center = 0,  ///< Project the segment onto the sector centre
    rightOverlap =
        1  ///< Project the segment on the overlap with the next sector
  };
  /// @brief Return the projector as a string
  static std::string toString(SectorProjector projector);
  /// @brief Define the ostream operator
  friend std::ostream& operator<<(std::ostream& ostr,
                                  SectorProjector projector) {
    return ostr << MuonExpandedSector::toString(projector);
  }
  friend std::ostream& operator<<(std::ostream& ostr,
                                  const MuonExpandedSector& sector) {
    return sector.print(ostr);
  }
  /// @brief Constructor of the expanded sector taking the
  ///        regular MS sector number and the projector
  /// @param msSector: Number of the ms reference sector [1-16]
  /// @param proj: Splitting of the sector to the overlap with the
  ///              left / right adjacent sector or the sector center
  explicit MuonExpandedSector(unsigned msSector, SectorProjector proj);
  /// @brief Constructor from an arbitrary phi angle. The angle is
  ///        assigned to the msSectors and then the expanded sector
  ///        is deduced
  /// @param phi: Angle from [-pi, pi]
  explicit MuonExpandedSector(double phi);
  /// @brief Constructor from a expanded sector number
  /// @param expandedSector: Raw expanded sector number
  explicit MuonExpandedSector(std::int8_t expandedSector);
  /// @brief Define the ordering operator
  bool operator<(const MuonExpandedSector& other) const;
  /// @brief Define the equal operator
  bool operator==(const MuonExpandedSector& other) const;
  /// @brief Define the unequal operator
  bool operator!=(const MuonExpandedSector& other) const;
  /// @brief Returns the ms sector corresponding to the
  ///        expanded sector.
  /// @note If the expanded sector is constructed with the
  ///       left / right overlap. The msSector number might be
  ///       the adjacent msSector
  unsigned msSector() const;
  /// @brief Returns the neighbouring msSector number constructed from
  ///        the primary sector and the sector overlap projector
  unsigned adjacentMsSector() const;
  /// @brief Returns the projector in the corresponding MS sector
  SectorProjector projector() const;
  /// @brief Returns the expanded sector number
  std::int8_t sector() const;
  /// @brief Returns the phi angle of the expanded sector
  double phi() const;
  /// @brief Returns the vector pointing radially along the sector plane
  Acts::Vector3 radialDir() const;
  /// @brief Returns the vector that is normal to the plane spanned
  ///        by the expanded sector
  Acts::Vector3 normalDir() const;
  /// @brief Returns whether the other expanded sector is a neighbour
  bool isNeighbour(const MuonExpandedSector& other) const;

 private:
  /// @brief Pipe the object to an ostream
  std::ostream& print(std::ostream& ostr) const;
  /// @brief the sector number stored
  std::int8_t m_sector{0};
};

}  // namespace ActsExamples

ACTS_OSTREAM_FORMATTER(ActsExamples::MuonStationIndex);
ACTS_OSTREAM_FORMATTER(ActsExamples::MuonLayerIndex);
ACTS_OSTREAM_FORMATTER(ActsExamples::MuonExpandedSector::SectorProjector);
ACTS_OSTREAM_FORMATTER(ActsExamples::MuonExpandedSector);
