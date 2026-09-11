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
#include <unordered_map>
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

/// @brief Data class representing a muon spectrometer global pattern.
class MuonGlobalPattern {
 public:
  using HitType = const MuonSpacePoint*;
  using StIndex = MuonStationIndex;
  using HitCollection = std::unordered_map<StIndex, std::vector<HitType>>;
  using BucketCollection =
      std::unordered_map<StIndex, std::vector<const MuonSpacePointBucket*>>;

  /// @brief Constructor consuming the hit collection per station
  MuonGlobalPattern(HitCollection&& hitPerStation,
                    BucketCollection&& bucketPerStation);
  MuonGlobalPattern() = delete;

  /// @brief Set the average theta of the pattern
  void setTheta(double theta) { m_theta = theta; }
  /// @brief Set the average phi of the pattern
  void setPhi(double phi) { m_phi = phi; }
  /// @brief Set the main sector of the pattern
  void setSector(std::int8_t sector) { m_sector = MuonExpandedSector{sector}; }
  /// @brief Set the number of precision layers in the pattern
  void setNPrecisionLayers(unsigned n) { m_nPrecisionLayers = n; }
  /// @brief Set the number of trigger layers in the pattern
  void setNTriggerLayers(unsigned n) { m_nTriggerLayers = n; }
  /// @brief Set the number of phi layers in the pattern
  void setNPhiLayers(unsigned n) { m_nPhiLayers = n; }
  /// @brief Set the mean over eta hits of the square of their residual
  ///        divided by the acceptance window from pattern finding
  void setMeanNormResidual2(double res) { m_meanNormResidual2 = res; }

  /// @brief Return the average global theta of the pattern
  double theta() const { return m_theta; }
  /// @brief Return the average global phi of the pattern
  double phi() const { return m_phi; }
  /// @brief Return the main sector where the pattern is located
  unsigned sector() const { return m_sector.msSector(); }
  /// @brief Return the associated sector of the pattern
  unsigned secondarySector() const { return m_sector.adjacentMsSector(); }
  /// @brief Return whether the pattern is located in the overlap region
  ///        between two sectors
  bool isSectorOverlap() const { return sector() != secondarySector(); }
  /// @brief Return the expanded sector of the pattern
  const MuonExpandedSector& expSector() const { return m_sector; }
  /// @brief Return the sector phi of the pattern. It is the central phi of
  ///        the sector or the value at the edge in case of overlap
  double sectorPhi() const;
  /// @brief Return the associated stations to the pattern
  std::vector<StIndex> getStations() const;
  /// @brief Return the pattern hits in the given station
  const std::vector<HitType>& hitsInStation(StIndex station) const;
  /// @brief Return the parent buckets of the pattern in the given station
  const std::vector<const MuonSpacePointBucket*>& bucketsInStation(
      StIndex station) const;
  /// @brief Return the number of precision layers in the pattern
  unsigned nPrecisionLayers() const { return m_nPrecisionLayers; }
  /// @brief Return the number of trigger layers in the pattern
  unsigned nTriggerLayers() const { return m_nTriggerLayers; }
  /// @brief Return the number of phi layers in the pattern
  unsigned nPhiLayers() const { return m_nPhiLayers; }
  /// @brief Return the mean over eta hits of the square of their residual
  ///        divided by the acceptance window from pattern finding
  double meanNormResidual2() const { return m_meanNormResidual2; }
  /// @brief Return the hits per station
  const HitCollection& hitsPerStation() const { return m_hitsInStation; }

  /// @brief The print-out operator
  friend std::ostream& operator<<(std::ostream& ostr,
                                  const MuonGlobalPattern& gp) {
    return gp.print(ostr);
  }
  /// @brief Equality operator. Compares hit content only and ignores
  ///        kinematics and quality fields
  bool operator==(const MuonGlobalPattern& other) const {
    return m_hitsInStation == other.m_hitsInStation;
  }

 private:
  std::ostream& print(std::ostream& ostr) const;

  double m_theta{0.};
  double m_phi{0.};
  unsigned m_nPrecisionLayers{0};
  unsigned m_nTriggerLayers{0};
  unsigned m_nPhiLayers{0};
  double m_meanNormResidual2{0.};

  /// Default is raw expanded sector 0 (MS sector 16, centre). Callers should
  /// setSector before use.
  MuonExpandedSector m_sector{static_cast<std::int8_t>(0)};

  HitCollection m_hitsInStation{};
  BucketCollection m_parentBuckets{};
};

using MuonGlobalPatternContainer = std::vector<MuonGlobalPattern>;

}  // namespace ActsExamples

ACTS_OSTREAM_FORMATTER(ActsExamples::MuonStationIndex);
ACTS_OSTREAM_FORMATTER(ActsExamples::MuonLayerIndex);
ACTS_OSTREAM_FORMATTER(ActsExamples::MuonExpandedSector::SectorProjector);
ACTS_OSTREAM_FORMATTER(ActsExamples::MuonExpandedSector);
ACTS_OSTREAM_FORMATTER(ActsExamples::MuonGlobalPattern);
