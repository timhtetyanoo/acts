// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Seeding/GlobalPatternFinder.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "Acts/Utilities/OstreamFormatter.hpp"
#include "ActsExamples/EventData/MuonSpacePoint.hpp"

#include <array>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

/** @brief Types plugging the example muon EDM into the Acts GlobalPatternFinder.
 *         Port of the Athena MuonR4::FastReco helpers
 * (MuonFastRecoHelpers/GlobalPatternFinderDefs.h), together with the parts of
 * ExpandedSector and MuonStationIndex they need. */
namespace ActsExamples {

/** @brief Helper functions to describe the expanded sector concept. The expanded
 *         sectors are based on the 16 fold symmetry of the MS, but also take
 * into account the overlap between 2 sectors.
 *
 *         The regular msSector is multiplied by 2 and then the sectorProjector
 * is added which can be either -1 to indicate that the overlap between the
 * current sector and the left sector is of interest, or 0 to indicate that the
 * sector center is of interest and finally 1 to indicate that the sector to the
 * right is of interest.
 */
class ExpandedSector {
 public:
  /** @brief Type of the index of the expanded sector */
  using Index_t = std::int8_t;
  /** @brief Enumeration to select the sector projection of the
   *         regular MS sector */
  enum class SectorProjector : std::int8_t {
    leftOverlap =
        -1,  /// Project the segment onto the overlap with the previous sector
    center = 0,  /// Project the segment onto the sector centre
    rightOverlap =
        1  /// Project the segment on the overlap with the next sector
  };
  /** @brief Return the projector as a string */
  static std::string toString(const SectorProjector proj);
  /** @brief Define the ostream operator */
  friend std::ostream& operator<<(std::ostream& ostr,
                                  const SectorProjector proj) {
    return ostr << (toString(proj));
  }
  friend std::ostream& operator<<(std::ostream& ostr,
                                  const ExpandedSector& sec) {
    return sec.toString(ostr);
  }
  /** @brief Constructor of the expanded sector taking the
   *         regular MS sector number and the projector
   * @param msSector: Number of the ms reference sector [1-16]
   * @param proj: Splitting of the sector to the overlap with the
   *              left / right adjacent sector or the sector center */
  explicit ExpandedSector(const unsigned msSector, const SectorProjector proj);
  /** @brief Constructor from an arbitrary phi angle. The angle is
   *         assigned to the msSectors and then the expanded sector
   *         is deduced
   *  @param phi: Angle from [-pi, pi] */
  explicit ExpandedSector(const double phi);
  /** @brief Constructor from a expanded sector number
   *  @param expSector: Raw expanded sector number */
  explicit ExpandedSector(const Index_t expSector);
  /** @brief Define the ordering operator */
  bool operator<(const ExpandedSector& other) const;
  /** @brief Define the equal operator */
  bool operator==(const ExpandedSector& other) const;
  /** @brief Define the unequal operator */
  bool operator!=(const ExpandedSector& other) const;
  /** @brief Returns the ms sector corresponding to the
   *         expanded sector.
   *  @note If the expanded sector is constructed with the
   *        left / right overlap. The msSector number might be
   *        the adjacent msSector */
  unsigned msSector() const;
  /** @brief Returns the neighbouring msSector number constructed from
   *         the primary sector and the sector overlap projector */
  unsigned adjacentMsSector() const;
  /** @brief Returns the projector in the corresponding MS sector */
  SectorProjector projector() const;
  /** @brief Returns the expanded sector number */
  Index_t sector() const;
  /** @brief Returns the phi angle of the expanded sector */
  double phi() const;
  /** @brief Returns the vector pointing radially along the sector plane  */
  Acts::Vector3 radialDir() const;
  /** @brief Returns the vector that is normal to the plane spanned
   *         by the expanded sector */
  Acts::Vector3 normalDir() const;
  /** @brief Returns true if the expanded sector is a neighbour of the other */
  bool isNeighbour(const ExpandedSector& other) const;
  /** @brief Check if a given phi is within the expanded sector */
  bool insideSector(const double phi) const;
  /** @brief Return the expanded sector (half) size */
  double sectorSize() const;

 private:
  /** @brief Pipe the object to an ostream  */
  std::ostream& toString(std::ostream& ostr) const;
  /** @brief the sector number stored */
  std::int8_t m_sector{0};
};

static_assert(Acts::Experimental::detail::SectorType<ExpandedSector>);

/** enum to classify the different station layers in the muon spectrometer */
enum class StIndex : std::int8_t {
  StUnknown = -1,
  BI,
  BM,
  BO,
  BE,
  EI,
  EM,
  EO,
  EE,
  StIndexMax
};

/** enum to classify the different layers in the muon spectrometer */
enum class LayerIndex : std::int8_t {
  LayerUnknown = -1,
  Inner,
  Middle,
  Outer,
  Extended,        /// EE
  BarrelExtended,  /// BEE
  LayerIndexMax
};

/** convert ChIndex into StIndex */
StIndex toStationIndex(MuonSpacePoint::MuonId::StationName index);

/** convert StIndex into LayerIndex */
LayerIndex toLayerIndex(StIndex index);

/** @brief Returns true if the station index points to a barrel chamber */
bool isBarrel(const StIndex index);

/** @brief Returns whether the uncalibrated spacepoint is a precision hit (Mdt, micromegas, stgc strips)
 *  @param hit: Reference to the uncalibrated space point */
bool isPrecisionHit(const MuonSpacePoint& hit);

/** @brief Base class for hit struct containing hit information. */
struct HitPayload {
  /** @brief Constructor with parameters
   *  @param sp The space point
   *  @param bucket The space point bucket
   *  @param localToGlobal The transformation from local to global coordinates
   *  @param locLayer The layer number in the sector frame
   *  @param station The station index */
  explicit HitPayload(const Acts::GeometryContext& gctx,
                      const MuonSpacePoint* sp,
                      const MuonSpacePointBucket* bucket,
                      const Acts::Transform3& localToGlobal,
                      const Acts::Surface* measSurface);
  /** @brief Retrieve the space point */
  const MuonSpacePoint* spacePoint() const { return underlyingSp; }
  /** @brief Retrieve the global position */
  const Acts::Vector3& globalPosition(const Acts::GeometryContext& gctx) const;
  /** @brief Retrieve the sensor direction */
  Acts::Vector3 globalSensorDirection(const Acts::GeometryContext& gctx) const;
  /** @brief Hit contribution contribution to the residual variance
             due to its intrinsic position uncertainty.
   *  @param contractionVector The contraction vector to compute the residual variance
   *  @param isProjected Whether the hit has been projected
   *  @return Residual variance contribution */
  double intrinsicVariance(const Acts::GeometryContext& gctx,
                           const Acts::Vector3& contractionVector) const;
  /** @brief Retrieve the phi variance of the hit */
  double phiVariance(const Acts::GeometryContext& gctx) const;
  /** @brief Returns whether the hit is a precision hit */
  bool isPrecision() const { return isPrecisionFlag; }

  /** @brief Global position */
  Acts::Vector3 position{Acts::Vector3::Zero()};
  /** @brief Pointer to the underlying hit */
  const MuonSpacePoint* underlyingSp{nullptr};
  /** @brief Pointer to the parent bucket */
  const MuonSpacePointBucket* bucket{nullptr};
  /** @brief Measurement surface of the hit. Athena fetches it from
   *         xAOD::muonSurface(spacePoint()->primaryMeasurement()) */
  const Acts::Surface* surface{nullptr};
  /** @brief Cached angular covariance [rad^2] of the hit in the phi angle */
  double phiCov{0.};
  /** @brief Strip angle when the strips are non-orthogonal */
  double stripAngle{0.};
  /** @brief Station index */
  StIndex station{toStationIndex(spacePoint()->id().msStation())};
  /** @brief Layer number in the sector frame */
  std::uint8_t locLayer{
      static_cast<std::uint8_t>(spacePoint()->id().detLayer())};
  /** @brief Is precision hit */
  bool isPrecisionFlag{isPrecisionHit(*spacePoint())};
  /** @brief Are the strips non-orthogonal */
  bool nonOrthogonalStrips{false};
  /** @brief Equal operator: it compares the underlying hit */
  bool operator==(const HitPayload& other) const;
};
static_assert(Acts::Experimental::detail::GlobPatFinderHit<HitPayload>);

class PatternTopology {
 public:
  /** @brief Type of the index of the station layer */
  using LayerIdx = std::uint8_t;
  /** @brief Type of the index of the group */
  using GroupIdx = std::uint8_t;
  /** @brief Number of groups in a pattern */
  static constexpr GroupIdx nGroups{
      static_cast<GroupIdx>(Acts::toUnderlying(StIndex::StIndexMax))};
  /** @brief Layer sorter */
  static bool layerSorter(const HitPayload& hit1, const HitPayload& hit2);
  /** @brief Return the station index */
  static GroupIdx groupIndex(const HitPayload& hit);
  /** @brief Check if two hits are in the same layer */
  static bool sameLayer(const HitPayload& hit1, const HitPayload& hit2);
};
static_assert(
    Acts::Experimental::detail::PatternTopology<PatternTopology, HitPayload>);

using GlobalPatternFinder_t =
    Acts::Experimental::detail::GlobalPatternFinder<HitPayload, ExpandedSector,
                                                    PatternTopology>;

using SearchTree_t = GlobalPatternFinder_t::SearchTree_t;

/** @brief Structure to hold the search tree data */
struct SearchTreeData {
  /** @brief Vector of strip hits */
  std::vector<HitPayload> stripPayloads;
  /** @brief The search tree */
  SearchTree_t tree;
};

struct SeedSelector {
  struct Config {
    bool seedFromMdt{false};
    double thetaSearchWindow{0.05};
    bool seedFromInner{false};
  };
  /** @brief Constructor */
  explicit SeedSelector(Config&& config);
  /** @brief Select a good seed */
  bool goodForSeeding(const HitPayload& hit) const;
  /** @brief Theta search window */
  double thetaSearchWindow(const HitPayload& hit1) const;
  std::vector<LayerIndex> m_seedinglayers{LayerIndex::Middle,
                                          LayerIndex::Outer};
  /** @brief Config */
  Config m_cfg;
};
static_assert(
    Acts::Experimental::detail::PatternSeedSelector<SeedSelector, HitPayload>);

struct OnlyPhiHitsProvider {
  using PatternState = GlobalPatternFinder_t::PatternState;
  using PhiHitsPerGroup =
      std::array<std::vector<HitPayload>, PatternTopology::nGroups>;
  /** @brief Get the phi-only hits compatible with the pattern */
  PhiHitsPerGroup getPhiOnlyHits(const PatternState& pattern,
                                 const Acts::GeometryContext& gctx) const;
  /** @brief Tracking geometry to look up the measurement surfaces. Athena reaches
   *         them through the space point's primary measurement */
  const Acts::TrackingGeometry* trackingGeometry{nullptr};
};
static_assert(Acts::Experimental::detail::OnlyPhiHitsProvider<
              OnlyPhiHitsProvider, HitPayload, PatternTopology,
              GlobalPatternFinder_t::PatternState>);

/** @brief Transformation from the bucket (sector) frame into the global frame.
 *         Replaces Athena's bucket->msSector()->localToGlobalTransform(gctx).
 * The bucket carries the transform of its first space point's surface into the
 * sector frame, so the global frame is reached as surfaceToGlobal *
 * surfaceToSector^-1.
 *  @param gctx: Geometry context
 *  @param trackingGeometry: Geometry holding the measurement surfaces
 *  @param bucket: Non-empty space point bucket */
Acts::Transform3 localToGlobalTransform(
    const Acts::GeometryContext& gctx,
    const Acts::TrackingGeometry& trackingGeometry,
    const MuonSpacePointBucket& bucket);

}  // namespace ActsExamples

ACTS_OSTREAM_FORMATTER(ActsExamples::ExpandedSector::SectorProjector);
