// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Units.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/EventData/MuonSpacePoint.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "ActsExamples/Framework/ProcessCode.hpp"
#include "ActsExamples/TrackFinding/GlobalPatternFinderDefs.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace ActsExamples {

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
    /** @brief Tracking geometry holding the muon measurement surfaces */
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry{};

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
  OnlyPhiHitsProvider m_onlyPhiProvider{};
  /** @brief Pointer to the actual global pattern finder */
  std::unique_ptr<GlobalPatternFinder_t> m_globPatFinder{};
};

}  // namespace ActsExamples
