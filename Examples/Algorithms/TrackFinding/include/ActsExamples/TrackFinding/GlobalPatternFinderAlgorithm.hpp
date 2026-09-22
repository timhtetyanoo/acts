// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Seeding/GlobalPatternFinder.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/OstreamFormatter.hpp"
#include "ActsExamples/EventData/MuonSpacePoint.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "ActsExamples/Framework/ProcessCode.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

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
                      const Acts::Transform3& localToGlobal);
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
  static PhiHitsPerGroup getPhiOnlyHits(const PatternState& pattern,
                                        const Acts::GeometryContext& gctx);
};
static_assert(Acts::Experimental::detail::OnlyPhiHitsProvider<
              OnlyPhiHitsProvider, HitPayload, PatternTopology,
              GlobalPatternFinder_t::PatternState>);

/** @brief Output global pattern. Slim version of the Athena GlobalPattern: it refers to the
 *         space points in the event store instead of the hit payloads, which
 * only live during the execution of the algorithm. */
struct MuonGlobalPattern {
  /** Hits of the pattern organized per station */
  std::array<std::vector<const MuonSpacePoint*>, PatternTopology::nGroups>
      hitsPerStation{};
  /** average global theta of the pattern */
  double theta{0.};
  /** average global phi of the pattern */
  double phi{0.};
  /** Expanded sector of the pattern */
  ExpandedSector::Index_t sector{0};
  /** Number of precision layers */
  unsigned nPrecisionLayers{0};
  /** Number of trigger layers */
  unsigned nTriggerLayers{0};
  /** Number of phi layers */
  unsigned nPhiLayers{0};
  /** Mean over eta hits of the square of their residual divided by acceptance
   * window from pattern finding */
  double meanNormResidual2{0.};
};
/** @brief Abrivation of the MuonGlobalPattern container type */
using MuonGlobalPatternContainer = std::vector<MuonGlobalPattern>;

/// @brief Algorithm performing global pattern recognition.
///
/// This algorithm performs global pattern recognition as the first step
/// of the Phase-2 fast reconstruction stage. It builds global patterns
/// of precision and non-precision hits using space-points created in
/// upstream algorithms. It first builds patterns in eta and then adds
/// compatible phi-only hits to the patterns. The resulting patterns are
/// written into store gate.
class GlobalPatternFinderAlgorithm final : public IAlgorithm {
 public:
  struct Config {
    /** @brief Keys of SpacePoint containers to read */
    std::string inSpacePoints{};
    /** @brief Write handle key for the output global patterns */
    std::string outPatterns{};

    /** @brief Toggle the utilization of MDT hits to build patterns */
    bool useMdtHits{true};
    /** @brief Toggle the seeding from MDT hits */
    bool seedFromMdt{false};
    /** @brief Activate the seeding from Inner station */
    bool seedFromInner{false};
    /** @brief Size of theta window [rad] to search for compatible hits with a seed, tailored to the target pt cutoff */
    double thetaSearchWindow{0.06};
    /** @brief Number of standard deviations to consider for residual acceptance */
    double nResidualSigma{3.};
    /** @brief Residual uncertainty to consider the hit as low confidence */
    double lowConfidenceResSigma{50.0};
    /** @brief Number of standard deviations to consider for phi compatibility veto. The residual will be used to determine the acceptance. */
    double nPhiSigma{5.};
    /** @brief Requirement on trigger layers in the bending direction to accept a pattern  */
    unsigned minTriggerLayers{2};
    /** @brief Requirement on precision layers in the bending direction to accept a pattern  */
    unsigned minPrecisionLayers{8};
    /** @brief Minimum number of phi layers required to accept a pattern */
    unsigned minPhiLayers{1};
    /** @brief Minimum number of layers in a station to be considered a good station */
    unsigned minStationLayers{4};
    /** @brief Quality cut on pattern'mean squared normalized residual. Set to a large value to disable the cut, e.g. 10. */
    double meanNormRes2Cut{3.5};
    /** @brief Maximum number of attempts to build a pattern from hits already used in existing patterns */
    unsigned maxSeedAttempts{3};
    /** @brief Maximum number of missed candidate hits in different measurement layers during pattern building allowed for a pattern branch before it is discarded */
    unsigned maxMissLayersInStation{3};
    /** @brief Minimum distance [mm] between two hits for being used to compute a reliable pattern line. Use the beamspot otherwise. */
    double minHitDistance4Line{200};
    /** @brief Beam spot radius */
    double beamSpotRadius{30. * Acts::UnitConstants::cm};
    /** @brief Beam spot length */
    double beamSpotLength{2. * Acts::UnitConstants::m};
  };

  explicit GlobalPatternFinderAlgorithm(
      const Config& cfg, std::unique_ptr<const Acts::Logger> logger = nullptr);

  ProcessCode execute(const AlgorithmContext& ctx) const override;

  /** @brief Const access to the config */
  const Config& config() const { return m_cfg; }

 private:
  using PatternResult = GlobalPatternFinder_t::OutputPattern;

  SearchTreeData constructTree(
      const Acts::GeometryContext& gctx,
      const MuonSpacePointContainer& spacepoints) const;
  /** @brief Method to convert a PatternState into a GlobalPattern object
   *  @param candidate: PatternState to be converted
   *  @return: Converted GlobalPattern */
  MuonGlobalPattern convertToPattern(const PatternResult& candidate) const;
  /** @brief Method to convert a vector of PatternStates into GlobalPattern objects
   *  @param candidates: PatternStates to be converted
   *  @return: Vector of converted GlobalPatterns */
  MuonGlobalPatternContainer convertToPattern(
      const std::vector<PatternResult>& candidates) const;

  Config m_cfg;

  /** @brief Keys of SpacePoint containers to read */
  ReadDataHandle<MuonSpacePointContainer> m_inSpacePoints{this,
                                                          "InSpacePoints"};
  /** @brief Write handle key for the output global patterns */
  WriteDataHandle<MuonGlobalPatternContainer> m_outPatterns{this,
                                                            "OutPatterns"};

  /** @brief Seed selector */
  std::unique_ptr<SeedSelector> m_seedSelector{};
  static constexpr OnlyPhiHitsProvider m_onlyPhiProvider{};
  /** @brief Pointer to the actual global pattern finder */
  std::unique_ptr<GlobalPatternFinder_t> m_globPatFinder{};
};

}  // namespace ActsExamples

ACTS_OSTREAM_FORMATTER(ActsExamples::ExpandedSector::SectorProjector);
