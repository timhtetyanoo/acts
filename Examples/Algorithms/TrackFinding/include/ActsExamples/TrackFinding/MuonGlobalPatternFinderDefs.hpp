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
#include "Acts/Seeding/detail/CompSpacePointAuxiliaries.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/EventData/MuonGlobalPatternFinderEvent.hpp"
#include "ActsExamples/EventData/MuonSpacePoint.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <vector>

/// @brief Internal data structures of the muon global pattern finder
namespace ActsExamples::MuonGlobalPatternFinderDefs {

/// @brief Components of the space point covariance array
enum class CovIdx : std::uint8_t {
  phiCov = Acts::toUnderlying(
      Acts::Experimental::detail::CompSpacePointAuxiliaries::ResidualIdx::
          nonBending),
  etaCov = Acts::toUnderlying(
      Acts::Experimental::detail::CompSpacePointAuxiliaries::ResidualIdx::
          bending),
  timeCov = Acts::toUnderlying(
      Acts::Experimental::detail::CompSpacePointAuxiliaries::ResidualIdx::time)
};

/// @brief Number of stations
constexpr std::size_t s_nStations =
    static_cast<std::size_t>(Acts::toUnderlying(MuonStationIndex::MaxVal));

/// @brief Converts a valid station index into an array index
constexpr std::size_t stationIdx(MuonStationIndex station) {
  assert(station != MuonStationIndex::UnDef &&
         station != MuonStationIndex::MaxVal);
  return static_cast<std::size_t>(Acts::toUnderlying(station));
}

/// @brief Configuration of the pattern finder. The defaults are the values
///        used in the Athena job configuration.
struct PatternFinderConfig {
  /// @brief Toggle the utilization of MDT hits to build patterns
  bool useMdtHits{true};
  /// @brief Toggle the seeding from MDT hits
  bool seedFromMdt{false};
  /// @brief Station layers from which the patterns are seeded
  std::vector<MuonLayerIndex> layerSeedings{MuonLayerIndex::Middle,
                                            MuonLayerIndex::Outer};
  /// @brief Size of theta window [rad] to search for compatible hits with a
  ///        seed, tailored to the target pt cutoff
  double thetaSearchWindow{0.06};
  /// @brief Number of standard deviations to consider for residual acceptance
  double nResidualSigma{3.};
  /// @brief Residual uncertainty to consider the hit as low confidence
  double lowConfidenceResSigma{50.};
  /// @brief Number of standard deviations to consider for phi acceptance
  double nPhiSigma{5.};
  /// @brief Minimum number of trigger layers in the bending direction
  unsigned minTriggerLayers{1u};
  /// @brief Minimum number of precision layers in the bending direction
  unsigned minPrecisionLayers{8u};
  /// @brief Minimum number of phi layers required to accept a pattern
  unsigned minPhiLayers{1u};
  /// @brief Minimum number of layers in a station to be a good station
  unsigned minStationLayers{4u};
  /// @brief Quality cut on the pattern mean squared normalized residual
  double meanNormRes2Cut{3.5};
  /// @brief Maximum number of attempts to build a pattern from hits already
  ///        used in existing patterns
  unsigned maxSeedAttempts{3u};
  /// @brief Maximum number of missed candidate hits in different measurement
  ///        layers in a station
  unsigned maxMissLayersInStation{3u};
  /// @brief Minimum distance between two hits to compute a reliable pattern
  ///        line. The beamspot is used otherwise.
  double minHitDistance4Line{200. * Acts::UnitConstants::mm};
  /// @brief Beamspot radius
  double beamSpotRadius{30. * Acts::UnitConstants::cm};
  /// @brief Beamspot length
  double beamSpotLength{2. * Acts::UnitConstants::m};
};

/// @brief Hit information cached for the pattern finding. It holds the global
///        position & sensor directions of the space point together with its
///        azimuthal covariance.
struct HitPayload {
  /// @brief Constructor
  /// @param sp: The space point
  /// @param bucket: The bucket holding the space point
  /// @param localToGlobal: Transform from the space point frame to global
  /// @param locLayer: The layer number in the bucket
  /// @param station: The station index
  explicit HitPayload(const MuonSpacePoint* sp,
                      const MuonSpacePointBucket* bucket,
                      const Acts::Transform3& localToGlobal,
                      std::uint8_t locLayer, MuonStationIndex station);

  /// @brief Hit contribution to the residual variance due to its intrinsic
  ///        position uncertainty
  /// @param contractionVector: The contraction vector to compute the residual
  ///        variance. If the hit is not projected, it is the unit residual
  ///        direction. Otherwise, it's J^T * residualDirection, where J is the
  ///        Jacobian of the projection
  /// @param isProjected: Whether the hit has been projected
  /// @return Residual variance contribution
  double residualVariance(const Acts::Vector3& contractionVector,
                          bool isProjected) const;

  /// @brief Global position
  Acts::Vector3 position{Acts::Vector3::Zero()};
  /// @brief Global sensor direction of the precision measurement
  Acts::Vector3 sensorDir{Acts::Vector3::Zero()};
  /// @brief For strip hits with phi: measurement direction independent of
  ///        sensorDir, which is the eta measurement direction if the strips
  ///        are orthogonal, the phi measurement direction otherwise.
  ///        For straw hits, the x component is repurposed to store the
  ///        transverse covariance of the drift radius.
  Acts::Vector3 secondaryMeasDir{Acts::Vector3::Zero()};
  /// @brief Pointer to the underlying space point
  const MuonSpacePoint* sp{nullptr};
  /// @brief Pointer to the parent bucket
  const MuonSpacePointBucket* bucket{nullptr};
  /// @brief Cached angular covariance [rad^2] of the hit in the phi angle
  double phiCov{0.};
  /// @brief Station index
  MuonStationIndex station{MuonStationIndex::UnDef};
  /// @brief Layer number in the bucket
  std::uint8_t locLayer{0u};
  /// @brief Is the hit a straw
  bool isStraw{false};
  /// @brief Is precision hit
  bool isPrecision{false};
  /// @brief Does the hit measure phi
  bool measuresPhi{false};
  /// @brief Does the hit measure eta
  bool measuresEta{false};
  /// @brief Are the strips non-orthogonal
  bool nonOrthogonalStrips{false};

  /// @brief Equal operator: it compares the underlying space point
  bool operator==(const HitPayload& other) const { return sp == other.sp; }
  /// @brief Arrow operator: it allows to access the underlying space point
  const MuonSpacePoint* operator->() const { return sp; }
  /// @brief Dereference operator: it allows to access the underlying space point
  const MuonSpacePoint& operator*() const { return *sp; }
};

/// @brief Small wrapper for candidate hits used to build patterns. This is
///        needed because the global layer number cannot be defined globally,
///        but it can be computed given a set of hits.
struct CandidateHit {
  /// @brief Pointer to the underlying hit
  const HitPayload* hit{nullptr};
  /// @brief Station index
  MuonStationIndex station{MuonStationIndex::UnDef};
  /// @brief Global measurement layer number
  std::uint8_t globLayer{0u};

  const HitPayload* operator->() const { return hit; }
  const HitPayload& operator*() const { return *hit; }
  const MuonSpacePoint* sp() const { return hit->sp; }
  bool operator==(const CandidateHit& other) const {
    return *hit == *other.hit;
  }
  bool operator==(const HitPayload& other) const { return *hit == other; }

  /// @brief Print the candidate hit
  void print(std::ostream& ostr) const;
  friend std::ostream& operator<<(std::ostream& ostr, const CandidateHit& c) {
    c.print(ostr);
    return ostr;
  }
};

/// @brief Possible outcomes of the pattern line compatibility test
enum class LineTestDecision : std::int8_t {
  /// @brief Test successful, add the hit to the pattern
  eAddHit,
  /// @brief Test successful with a pattern hit on the same layer, branch the
  ///        pattern
  eBranchPattern,
  /// @brief Test failed, discard the hit
  eRejectHit
};

/// @brief Result of the line compatibility test
struct LineTestRes {
  /// @brief Distance of the test hit to the pattern line
  double residual{0.};
  /// @brief Uncertainty of the residual
  double sigma{0.};
  /// @brief Decision of the test
  LineTestDecision result{LineTestDecision::eRejectHit};
};

/// @brief Pattern state object storing the pattern information during its
///        construction
struct PatternState {
  /// @brief Constructor taking the seed information
  /// @param seed: Seed hit
  /// @param expSector: Raw **expanded** sector number
  /// @param cfg: Pointer to the configuration object
  /// @param logger: Pointer to the logger
  explicit PatternState(const CandidateHit& seed, std::int8_t expSector,
                        const PatternFinderConfig* cfg,
                        const Acts::Logger* logger);
  PatternState(PatternState&& other) noexcept = default;
  PatternState& operator=(PatternState&& other) noexcept = default;
  PatternState(const PatternState& other) = default;
  PatternState& operator=(const PatternState& other) = default;
  ~PatternState() = default;

  /// @brief Add a hit to the pattern and update the internal state
  /// @param hit: Hit to be added
  /// @param residual: Residual of the hit
  /// @param resSigma: Residual uncertainty of the hit
  void addHit(const CandidateHit& hit, double residual, double resSigma);
  /// @brief Overwrite the last inserted hit with a new one on the same layer
  /// @param newHit: New hit to replace the last inserted one
  /// @param newResidual: Residual of the new hit
  /// @param newResSigma: Residual uncertainty of the new hit
  void overWriteHit(const CandidateHit& newHit, double newResidual,
                    double newResSigma);
  /// @brief Check the line compatibility of a test hit against the pattern. The
  ///        line is updated if the test hit is on a new layer.
  /// @param testHit: Test hit
  /// @param beamSpot: Beam spot position, needed to update the pattern line
  /// @return Result of the test holding the residual and its uncertainty
  LineTestRes checkLineComp(const CandidateHit& testHit,
                            const Acts::Vector3& beamSpot);
  /// @brief Compute the residual of a test hit against the pattern line
  /// @param testHit: Test hit
  /// @return Test result holding the residual and its uncertainty. The
  ///         decision is set by checkLineComp.
  LineTestRes computeLineResidual(const CandidateHit& testHit) const;

  /// @brief Project a hit position onto the bending plane where the pattern is
  ///        defined. The hit is moved along its sensor direction.
  /// @param hit: Hit whose position is to be projected
  Acts::Vector3 projToPhiPlane(const HitPayload& hit) const;
  /// @brief Check the phi compatibility of a test hit with the pattern. If the
  ///        pattern has phi measurements, the hit must be within nPhiSigma of
  ///        the pattern phi. Otherwise, it must be inside the pattern sector(s)
  /// @param hit: Hit to be checked
  bool isPhiCompatible(const HitPayload& hit) const;
  /// @brief Check whether a hit is present in the pattern
  bool isInPattern(const HitPayload& hit) const;
  /// @brief Move the line anchor hit given a reference hit. The anchor is
  ///        defined as the closest hit in the closest station to the reference
  ///        hit.
  void moveLineAnchorHit(const CandidateHit& refHit);
  /// @brief Update the line parameters based on the anchor & last inserted hit
  /// @param beamSpot: Beam spot position, needed when the hits are too close
  void updateLineParameters(const Acts::Vector3& beamSpot);
  /// @brief Update the pattern phi and the bending plane normal
  void updatePatternPhi();
  /// @brief Return the mean normalized residual squared
  double getMeanResidual2() const;
  /// @brief Return the number of layers in the bending coordinate
  std::uint8_t nBendingLayers() const;
  /// @brief Return the number of stations
  /// @param onlyGoodStations: Count only stations having at least
  ///        minStationLayers hits
  std::uint8_t nStations(bool onlyGoodStations) const;
  /// @brief Return the buckets associated with the pattern hits
  std::vector<const MuonSpacePointBucket*> getParentBuckets() const;
  /// @brief Print the pattern state
  void print(std::ostream& ostr, bool detailedPrint) const;

  /// @brief Pointer to the configuration
  const PatternFinderConfig* cfg{nullptr};
  /// @brief Pointer to the logger
  const Acts::Logger* m_logger{nullptr};
  /// @brief Last inserted hit
  CandidateHit lastInsertedHit{};
  /// @brief Last hit in the second-to-last layer
  CandidateHit prevLayerHit{};
  /// @brief Line anchor hit
  CandidateHit lineAnchorHit{};
  /// @brief Normal vector to the bending plane where the pattern lies
  Acts::Vector3 bendPlaneNorm{Acts::Vector3::Zero()};
  /// @brief Position and direction of the pattern line, both within the
  ///        bending plane of the pattern
  Acts::Vector3 linePos{Acts::Vector3::Zero()};
  Acts::Vector3 lineDir{Acts::Vector3::Zero()};
  /// @brief Distance between the two points defining the pattern line
  double leverArm{0.};
  /// @brief Sum (mean once finalized) over eta hits of the squared residual
  ///        divided by its uncertainty
  double meanNormResidual2{0.};
  /// @brief Residual & residual uncertainty of the last inserted hit
  double lastResidual{0.};
  double lastResSigma{0.};
  /// @brief Pattern phi, i.e. the phi of the bending plane
  double patPhi{0.};
  /// @brief Pattern theta, i.e. the theta of the seed hit
  double patTheta{0.};
  /// @brief Covariance of the pattern phi
  double patPhiCov{0.};
  /// @brief **Expanded** MS sector
  MuonExpandedSector expSect{static_cast<std::int8_t>(0)};
  /// @brief Counts of precision / trigger / phi layers
  std::uint8_t nPrecisionLayers{0u};
  std::uint8_t nTriggerLayers{0u};
  std::uint8_t nPhiLayers{0u};
  /// @brief Whether the pattern has been finalized
  bool isFinalized{false};
  /// @brief Whether the pattern overlaps with a better one
  bool isOverlap{false};
  /// @brief Whether the beamspot was used to compute the line parameters
  bool useBeamspot{false};
  /// @brief Whether the line has to be updated when a hit in a new layer is
  ///        found
  bool needLineUpdate{false};
  /// @brief Hits per station. A pattern is determined by its hits
  std::array<std::vector<CandidateHit>, s_nStations> hitsPerStation{};
  /// @brief Phi-only hits
  std::vector<HitPayload> phiOnlyHits{};

  /// @brief Patterns are identical if they have the same hit content, but the
  ///        comparison is too expensive to be done implicitly
  bool operator==(const PatternState& other) const = delete;

 private:
  const Acts::Logger& logger() const { return *m_logger; }
};

/// @brief A view of the pattern state for printing purposes
struct PatternPrintView {
  const PatternState& pat;
  bool isDetailed{false};
  friend std::ostream& operator<<(std::ostream& ostr,
                                  const PatternPrintView& v) {
    v.pat.print(ostr, v.isDetailed);
    return ostr;
  }
};
/// @brief Print the pattern state with brief information
inline PatternPrintView brief(const PatternState& pat) {
  return {pat, false};
}
/// @brief Print the pattern state with detailed information
inline PatternPrintView detailed(const PatternState& pat) {
  return {pat, true};
}

}  // namespace ActsExamples::MuonGlobalPatternFinderDefs
