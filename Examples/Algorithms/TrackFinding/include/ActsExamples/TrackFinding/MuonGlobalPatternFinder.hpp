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
#include "Acts/Utilities/KDTree.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/EventData/MuonGlobalPatternFinderEvent.hpp"
#include "ActsExamples/EventData/MuonSpacePoint.hpp"
#include "ActsExamples/TrackFinding/MuonGlobalPatternFinderDefs.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace ActsExamples {

/// @brief Standalone module performing the global pattern recognition in the
///        muon spectrometer, ported from the Athena
///        MuonR4::FastReco::GlobalPatternFinder. It builds global patterns of
///        precision and non-precision hits from the space points of the event.
///        It first builds patterns in eta and then adds compatible phi-only
///        hits to the patterns.
class MuonGlobalPatternFinder {
 public:
  using HitPayload = MuonGlobalPatternFinderDefs::HitPayload;
  using CandidateHit = MuonGlobalPatternFinderDefs::CandidateHit;
  using PatternState = MuonGlobalPatternFinderDefs::PatternState;
  using PatternStateVec = std::vector<PatternState>;

  /// @brief Configuration of the pattern finder
  struct Config : public MuonGlobalPatternFinderDefs::PatternFinderConfig {
    /// @brief Transform from the space point frame of a bucket to the global
    ///        frame
    std::function<Acts::Transform3(const Acts::GeometryContext&,
                                   const MuonSpacePointBucket&)>
        localToGlobal{};
  };

  /// @brief Coordinates of the search tree
  enum class SeedCoords : std::uint8_t {
    /// @brief **Expanded** sector of the associated spectrometer sector
    eSector,
    /// @brief Global theta
    eTheta
  };
  /// @brief Search tree ordering the hits in expanded sector & theta
  using SearchTree_t =
      Acts::KDTree<2, const HitPayload*, double, std::array, 15>;
  /// @brief Hit payloads together with the search tree pointing to them. The
  ///        object must not be moved once constructed, as the tree nodes refer
  ///        to the tree's internal element vector.
  struct SearchTreeData {
    /// @brief Payloads of the hits measuring eta
    std::vector<HitPayload> hitPayloads;
    /// @brief The search tree
    SearchTree_t tree;
  };
  /// @brief Logical measurement layer ordering of two hits
  enum class LayerOrdering : std::int8_t {
    eSameLayer,
    eLowerLayer,
    eHigherLayer
  };

  /// @brief Constructor
  /// @param config: Configuration of the pattern finder
  /// @param logger: Logger
  explicit MuonGlobalPatternFinder(Config config,
                                   std::unique_ptr<const Acts::Logger> logger);

  /// @brief Return the configuration
  const Config& config() const { return m_cfg; }

  /// @brief Find the global patterns of the event: build the search tree, find
  ///        the patterns in eta, attach the compatible phi-only hits and
  ///        convert the patterns.
  /// @param gctx: Geometry context
  /// @param spacePoints: Space point buckets of the event. The patterns refer
  ///        to them, so they must outlive the patterns.
  MuonGlobalPatternContainer findPatterns(
      const Acts::GeometryContext& gctx,
      const MuonSpacePointContainer& spacePoints) const;

  /// @note The following building blocks are public for unit testing

  /// @brief Construct the search tree from the space point buckets. The hits
  ///        are duplicated in the neighbouring expanded sectors if they are
  ///        close to the sector borders. Phi-only hits are not added.
  /// @param gctx: Geometry context
  /// @param spacePoints: Space point buckets of the event
  SearchTreeData constructTree(const Acts::GeometryContext& gctx,
                               const MuonSpacePointContainer& spacePoints) const;
  /// @brief Build the patterns in the bending plane. Every hit in the seeding
  ///        layers is used as seed to collect compatible hits within the
  ///        theta window, which are then tested layer by layer, first outwards
  ///        and then inwards from the seed.
  /// @param orderedSpacepoints: Search tree with the hits
  /// @return Patterns passing the quality cuts without overlaps
  PatternStateVec findPatternsInEta(
      const SearchTree_t& orderedSpacepoints) const;
  /// @brief Test a hit against the active patterns built from the same seed.
  ///        Patterns are branched when they are compatible with multiple hits
  ///        on the same layer and pruned when they are worse than a pattern
  ///        sharing the same last hit.
  /// @param startPatterns: Active patterns to be extended, cleared at the end
  /// @param endPatterns: Surviving patterns after testing the hit
  /// @param testHit: Hit to be tested against the patterns
  /// @param beamSpot: Beam spot position, needed when the pattern line cannot
  ///        be reliably defined from the pattern hits
  void extendPatterns(PatternStateVec& startPatterns,
                      PatternStateVec& endPatterns, const CandidateHit& testHit,
                      const Acts::Vector3& beamSpot) const;
  /// @brief Remove overlapping patterns keeping the best ones
  /// @param toResolve: Patterns to resolve. The surviving patterns are moved
  ///        out of it
  /// @return Patterns without overlaps
  PatternStateVec resolveOverlaps(PatternStateVec& toResolve) const;
  /// @brief Add the compatible phi-only hits of the parent buckets to the
  ///        patterns. Patterns with less than minPhiLayers phi layers are
  ///        removed.
  /// @param gctx: Geometry context
  /// @param patterns: Patterns to which the phi-only hits are added
  void addPhiOnlyHits(const Acts::GeometryContext& gctx,
                      PatternStateVec& patterns) const;
  /// @brief Convert a pattern state into a global pattern
  MuonGlobalPattern convertToPattern(const PatternState& candidate) const;
  /// @brief Convert pattern states into global patterns
  MuonGlobalPatternContainer convertToPattern(
      const PatternStateVec& candidates) const;
  /// @brief Check whether a pattern passes the quality cuts
  bool passPatternCuts(const PatternState& pat) const;
  /// @brief Return whether pattern a is better than pattern b
  static bool isBetter(const PatternState& a, const PatternState& b);
  /// @brief Return the logical measurement layer ordering of two hits
  static LayerOrdering checkLayerOrdering(const HitPayload& hit1,
                                          const HitPayload& hit2);

 private:
  const Acts::Logger& logger() const { return *m_logger; }

  Config m_cfg;
  std::unique_ptr<const Acts::Logger> m_logger;
};

}  // namespace ActsExamples
