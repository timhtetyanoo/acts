// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFinding/MuonGlobalPatternFinder.hpp"

#include "Acts/Definitions/Common.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/Surfaces/detail/LineHelper.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "Acts/Utilities/StringHelpers.hpp"
#include "Acts/Utilities/VectorHelpers.hpp"
#include "Acts/Utilities/detail/periodic.hpp"
#include "ActsExamples/TrackFinding/MuonGlobalPatternFinderUtils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <format>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

using namespace Acts::UnitLiterals;
using Acts::VectorHelpers::perp;
using Acts::VectorHelpers::phi;

namespace ActsExamples {

using MuonGlobalPatternFinderDefs::brief;
using MuonGlobalPatternFinderDefs::CovIdx;
using MuonGlobalPatternFinderDefs::detailed;
using MuonGlobalPatternFinderDefs::LineTestDecision;
using MuonGlobalPatternFinderDefs::LineTestRes;
using MuonGlobalPatternFinderDefs::s_nStations;
using MuonGlobalPatternFinderDefs::stationIdx;

namespace {

constexpr auto s_thetaIdx =
    Acts::toUnderlying(MuonGlobalPatternFinder::SeedCoords::eTheta);
constexpr auto s_sectorIdx =
    Acts::toUnderlying(MuonGlobalPatternFinder::SeedCoords::eSector);

}  // namespace

MuonGlobalPatternContainer MuonGlobalPatternFinder::findPatterns(
    const Acts::GeometryContext& gctx,
    const MuonSpacePointContainer& spacePoints) const {
  // The patterns refer to the hit payloads only during the pattern finding
  const SearchTreeData treeData = constructTree(gctx, spacePoints);
  PatternStateVec patterns = findPatternsInEta(treeData.tree);
  addPhiOnlyHits(gctx, patterns);
  ACTS_DEBUG("findPatterns() Found " << patterns.size() << " patterns.");
  return convertToPattern(patterns);
}

void MuonGlobalPatternFinder::addPhiOnlyHits(const Acts::GeometryContext& gctx,
                                             PatternStateVec& patterns) const {
  /// @brief Define the pattern line in a station. The two eta hits on the
  ///        outermost layers of the station define the line. If they are too
  ///        close, the closest hit in another station is used as anchor.
  /// @return Whether the line could be defined without the beamspot
  auto computePatternLineInStation = [](PatternState& pat,
                                        MuonStationIndex station) {
    const std::vector<CandidateHit>& stationHits =
        pat.hitsPerStation[stationIdx(station)];
    if (stationHits.empty()) {
      return false;
    }
    // useBeamspot flags whether the pattern line could not be determined
    pat.useBeamspot = true;
    if (stationHits.size() > 1u) {
      const auto [minIt, maxIt] =
          std::ranges::minmax_element(stationHits, {}, &CandidateHit::globLayer);
      pat.lineAnchorHit = *minIt;
      pat.lastInsertedHit = *maxIt;
      // @note Deviation from Athena: the line update is forced. Otherwise, it
      //       is skipped if the line was already updated after the last
      //       inserted hit, leaving useBeamspot set.
      pat.needLineUpdate = true;
      pat.updateLineParameters(Acts::Vector3::Zero());
    }
    if (pat.useBeamspot) {
      // Only one eta hit or too close hits: anchor the line in another station
      pat.moveLineAnchorHit(stationHits.front());
      const Acts::Vector3 anchorPos = pat.projToPhiPlane(*pat.lineAnchorHit);
      pat.lastInsertedHit = *std::ranges::max_element(
          stationHits, {}, [&pat, &anchorPos](const CandidateHit& c) {
            return (pat.projToPhiPlane(*c) - anchorPos).norm();
          });
      pat.needLineUpdate = true;
      pat.updateLineParameters(Acts::Vector3::Zero());
    }
    return !pat.useBeamspot;
  };

  PatternStateVec survivingPatterns{};
  survivingPatterns.reserve(patterns.size());
  for (PatternState& pat : patterns) {
    ACTS_VERBOSE("addPhiOnlyHits() Search for phi-only hits for pattern: "
                 << brief(pat));
    std::optional<MuonStationIndex> patternLineStation{};

    const auto projOntoPhiPlane = [&pat](const Acts::Vector3& pos) {
      return Acts::Vector3{pos - pos.dot(pat.bendPlaneNorm) * pat.bendPlaneNorm};
    };

    for (const MuonSpacePointBucket* bucket : pat.getParentBuckets()) {
      const Acts::Transform3 localToGlobal = m_cfg.localToGlobal(gctx, *bucket);
      const MuonStationIndex station =
          stationIndex(bucket->front().id().msStation());
      const std::vector<std::uint8_t> layers =
          MuonGlobalPatternFinderUtils::layerNumbers(*bucket);

      for (std::size_t i = 0u; i < bucket->size(); ++i) {
        const MuonSpacePoint& hit = (*bucket)[i];
        if (hit.id().measuresEta()) {
          continue;
        }
        ACTS_VERBOSE("addPhiOnlyHits() *** Test phi-only hit " << hit);

        // Reject hits from a layer that already contains a phi hit
        const std::uint8_t layNum = layers[i];
        const std::vector<CandidateHit>& stationHits =
            pat.hitsPerStation[stationIdx(station)];
        const bool layerHasPhi =
            std::ranges::any_of(stationHits,
                                [bucket, layNum](const CandidateHit& h) {
                                  return h->measuresPhi &&
                                         h->bucket == bucket &&
                                         h->locLayer == layNum;
                                }) ||
            std::ranges::any_of(pat.phiOnlyHits,
                                [bucket, layNum](const HitPayload& h) {
                                  return h.bucket == bucket &&
                                         h.locLayer == layNum;
                                });
        if (layerHasPhi) {
          ACTS_VERBOSE("addPhiOnlyHits() The pattern already has a phi hit in "
                       "the same layer - skip hit.");
          continue;
        }

        HitPayload newHit{&hit, bucket, localToGlobal, layNum, station};
        if (!pat.isPhiCompatible(newHit)) {
          ACTS_VERBOSE("addPhiOnlyHits() Phi-only hit not compatible.");
          continue;
        }
        if (!patternLineStation.has_value() || *patternLineStation != station) {
          if (!computePatternLineInStation(pat, station)) {
            ACTS_VERBOSE("addPhiOnlyHits() Invalid projection model for station "
                         << station << " - skip hit.");
            continue;
          }
          patternLineStation = station;
        }

        // Check that the pattern line crosses the strip along its length
        // @note Deviation from Athena: the payload of phi-only hits has no
        //       sensor direction, so the strip direction is taken from the
        //       space point.
        const Acts::Vector3 stripDir =
            localToGlobal.linear() * hit.sensorDirection();
        const double stripHalfLength =
            std::sqrt(hit.covariance()[Acts::toUnderlying(CovIdx::etaCov)]);
        const Acts::Vector3 stripLow =
            projOntoPhiPlane(newHit.position - stripHalfLength * stripDir);
        const Acts::Vector3 stripHigh =
            projOntoPhiPlane(newHit.position + stripHalfLength * stripDir);
        const double stripProjLength = (stripHigh - stripLow).norm();
        const double stripIntersect =
            Acts::detail::LineHelper::lineIntersect<3>(
                pat.linePos, pat.lineDir, stripLow,
                Acts::Vector3{(stripHigh - stripLow).normalized()})
                .pathLength();
        ACTS_VERBOSE("addPhiOnlyHits() Intersect distance from lower strip "
                     "edge: "
                     << stripIntersect
                     << ", projected strip length: " << stripProjLength);

        constexpr double margin = 10._mm;
        if (!(stripIntersect >= -margin &&
              stripIntersect <= stripProjLength + margin)) {
          ACTS_VERBOSE("addPhiOnlyHits() The pattern falls outside the strip "
                       "in eta - skip hit.");
          continue;
        }
        pat.phiOnlyHits.push_back(std::move(newHit));
        ++pat.nPhiLayers;
        pat.updatePatternPhi();
      }
    }
    if (pat.nPhiLayers < m_cfg.minPhiLayers) {
      ACTS_VERBOSE("addPhiOnlyHits() Pattern "
                   << detailed(pat) << " has only "
                   << static_cast<int>(pat.nPhiLayers)
                   << " phi layers, below the minimum - reject pattern.");
      continue;
    }
    survivingPatterns.push_back(std::move(pat));
  }
  std::swap(patterns, survivingPatterns);
}

MuonGlobalPattern MuonGlobalPatternFinder::convertToPattern(
    const PatternState& candidate) const {
  MuonGlobalPattern::HitCollection hitPerStation{};
  MuonGlobalPattern::BucketCollection bucketPerStation{};
  // Add the eta hits
  for (std::size_t st = 0u; st < s_nStations; ++st) {
    const std::vector<CandidateHit>& hits = candidate.hitsPerStation[st];
    if (hits.empty()) {
      continue;
    }
    const auto station =
        static_cast<MuonStationIndex>(static_cast<std::int8_t>(st));
    auto& outHits = hitPerStation[station];
    auto& outBuckets = bucketPerStation[station];
    outHits.reserve(hits.size());
    for (const CandidateHit& hit : hits) {
      outHits.push_back(hit.sp());
      if (std::ranges::find(outBuckets, hit->bucket) == outBuckets.end()) {
        outBuckets.push_back(hit->bucket);
      }
    }
  }
  // Add the phi-only hits
  for (const HitPayload& hit : candidate.phiOnlyHits) {
    hitPerStation[hit.station].push_back(hit.sp);
  }
  MuonGlobalPattern pattern{std::move(hitPerStation),
                            std::move(bucketPerStation)};
  pattern.setTheta(candidate.patTheta);
  pattern.setPhi(candidate.patPhi);
  pattern.setSector(candidate.expSect.sector());
  pattern.setNPrecisionLayers(candidate.nPrecisionLayers);
  pattern.setNTriggerLayers(candidate.nTriggerLayers);
  pattern.setNPhiLayers(candidate.nPhiLayers);
  pattern.setMeanNormResidual2(candidate.getMeanResidual2());
  return pattern;
}

MuonGlobalPatternContainer MuonGlobalPatternFinder::convertToPattern(
    const PatternStateVec& candidates) const {
  MuonGlobalPatternContainer patterns{};
  patterns.reserve(candidates.size());
  std::ranges::transform(
      candidates, std::back_inserter(patterns),
      [this](const PatternState& c) { return convertToPattern(c); });
  return patterns;
}

MuonGlobalPatternFinder::MuonGlobalPatternFinder(
    Config config, std::unique_ptr<const Acts::Logger> logger)
    : m_cfg{std::move(config)}, m_logger{std::move(logger)} {
  if (!m_cfg.localToGlobal) {
    throw std::invalid_argument(
        "MuonGlobalPatternFinder: the local to global transform is not set");
  }
  if (!m_logger) {
    throw std::invalid_argument("MuonGlobalPatternFinder: missing logger");
  }
}

MuonGlobalPatternFinder::SearchTreeData MuonGlobalPatternFinder::constructTree(
    const Acts::GeometryContext& gctx,
    const MuonSpacePointContainer& spacePoints) const {
  using enum MuonExpandedSector::SectorProjector;

  std::vector<HitPayload> hitPayloads{};
  // Reserve the full capacity upfront, as the tree refers to the payloads by
  // pointer
  std::size_t totalHits{0u};
  for (const MuonSpacePointBucket& bucket : spacePoints) {
    totalHits += bucket.size();
  }
  hitPayloads.reserve(totalHits);

  for (const MuonSpacePointBucket& bucket : spacePoints) {
    if (bucket.empty()) {
      continue;
    }
    const MuonStationIndex station =
        stationIndex(bucket.front().id().msStation());
    if (station == MuonStationIndex::UnDef) {
      ACTS_WARNING("constructTree() Skip bucket with undefined station "
                   << bucket.front().id());
      continue;
    }
    const Acts::Transform3 localToGlobal = m_cfg.localToGlobal(gctx, bucket);
    const std::vector<std::uint8_t> layers =
        MuonGlobalPatternFinderUtils::layerNumbers(bucket);

    for (std::size_t i = 0u; i < bucket.size(); ++i) {
      const MuonSpacePoint& sp = bucket[i];
      // Ignore phi-only hits and MDT hits if desired
      if (!sp.id().measuresEta() || (!m_cfg.useMdtHits && sp.isStraw())) {
        continue;
      }
      const HitPayload& newHit = hitPayloads.emplace_back(
          &sp, &bucket, localToGlobal, layers[i], station);
      ACTS_VERBOSE("constructTree() Building hit from "
                   << sp << "\nPhiCov: " << newHit.phiCov << ", Pos: "
                   << Acts::toString(newHit.position)
                   << ", SensorDir: " << Acts::toString(newHit.sensorDir));
    }
  }

  SearchTree_t::vector_t treeData{};
  treeData.reserve(3u * hitPayloads.size());
  for (const HitPayload& hit : hitPayloads) {
    const Acts::Vector3& pos = hit.position;
    // The expanded sector of the hit position is only meaningful if it
    // measures phi
    std::optional<MuonExpandedSector> hitExpSector{};
    if (hit.measuresPhi) {
      hitExpSector.emplace(phi(pos));
    }
    // Duplicate the hit in the neighbouring sectors if it is close to the
    // sector border. This ensures that patterns crossing the sector borders
    // can be found.
    for (const auto proj : {leftOverlap, center, rightOverlap}) {
      const MuonExpandedSector expSect{static_cast<unsigned>(hit->id().sector()),
                                       proj};
      if (proj != center && hitExpSector.has_value() &&
          expSect != *hitExpSector) {
        ACTS_VERBOSE("constructTree() Hit with " << *hitExpSector
                                                 << " is not compatible with "
                                                 << expSect);
        continue;
      }
      // Project the hit onto the plane along the sector radial direction. This
      // removes the bias of the hit displacement in the phi direction.
      const Acts::Vector3 planeNormal = expSect.normalDir();
      const double projR =
          hit.measuresPhi ? perp(pos)
                          : perp(Acts::Vector3{
                                pos - pos.dot(planeNormal) * planeNormal});

      SearchTree_t::coordinate_t coords{};
      coords[s_thetaIdx] = std::atan2(projR, pos.z());
      coords[s_sectorIdx] = expSect.sector();
      ACTS_VERBOSE("constructTree() Add hit: Z: "
                   << pos.z() << ", R: " << perp(pos) << ", ProjR: " << projR
                   << ", Phi: " << phi(pos) / 1._degree
                   << ", SectorPhi: " << expSect.phi() / 1._degree
                   << " to the search tree");
      treeData.emplace_back(coords, &hit);
    }
  }
  ACTS_DEBUG("constructTree() Create a new tree with "
             << treeData.size() << " entries and " << hitPayloads.size()
             << " hits.");
  return SearchTreeData{std::move(hitPayloads),
                        SearchTree_t{std::move(treeData)}};
}

MuonGlobalPatternFinder::PatternStateVec
MuonGlobalPatternFinder::findPatternsInEta(
    const SearchTree_t& orderedSpacepoints) const {
  using enum LayerOrdering;
  const unsigned minLayers =
      m_cfg.minTriggerLayers + m_cfg.minPrecisionLayers;

  std::vector<CandidateHit> candidateHits{};
  candidateHits.reserve(100);
  // Two buffers of active patterns to avoid reallocations
  PatternStateVec startPatternBuff{};
  PatternStateVec endPatternBuff{};
  startPatternBuff.reserve(10);
  endPatternBuff.reserve(10);

  /// @todo Retrieve the beamspot if desired
  const Acts::Vector3 beamSpot{Acts::Vector3::Zero()};

  PatternStateVec outPatterns{};
  outPatterns.reserve(10);

  /// @brief Count the existing patterns containing a hit
  auto countPatterns = [this](const PatternStateVec& patterns,
                              const HitPayload& hit,
                              const SearchTree_t::coordinate_t& coords) {
    const MuonExpandedSector hitSector{
        static_cast<std::int8_t>(coords[s_sectorIdx])};
    return static_cast<std::size_t>(
        std::ranges::count_if(patterns, [&](const PatternState& pattern) {
          if (std::abs(pattern.patTheta - coords[s_thetaIdx]) >
                  2. * m_cfg.thetaSearchWindow ||
              !pattern.expSect.isNeighbour(hitSector)) {
            return false;
          }
          return pattern.isInPattern(hit);
        }));
  };

  for (const MuonLayerIndex seedingLayer : m_cfg.layerSeedings) {
    // Try to build a pattern starting from every hit in the tree
    for (const auto& [seedCoords, seedPtr] : orderedSpacepoints) {
      const HitPayload& seed = *seedPtr;
      // Check that the seed is in the current seeding layer and whether
      // seeding from MDT hits is enabled
      const MuonLayerIndex seedLayer = layerIndex(seed.station);
      if (seedLayer != seedingLayer || (seed.isStraw && !m_cfg.seedFromMdt)) {
        continue;
      }
      ACTS_VERBOSE("findPatternsInEta() New seed hit "
                   << *seed.sp << ", sector: " << seedCoords[s_sectorIdx]
                   << ", theta: " << seedCoords[s_thetaIdx]);

      // Check how many existing patterns contain this hit
      std::size_t nExistingPatterns =
          countPatterns(outPatterns, seed, seedCoords);
      if (nExistingPatterns >= m_cfg.maxSeedAttempts) {
        // Try first to resolve overlaps and recount
        outPatterns = resolveOverlaps(outPatterns);
        nExistingPatterns = countPatterns(outPatterns, seed, seedCoords);
        if (nExistingPatterns >= m_cfg.maxSeedAttempts) {
          ACTS_VERBOSE("findPatternsInEta() Seed has already been used in "
                       << nExistingPatterns
                       << " patterns, which is above the limit - skip.");
          continue;
        }
      }

      // Search hits in the same **expanded** sector
      SearchTree_t::range_t selectRange{};
      selectRange[s_sectorIdx].shrink(seedCoords[s_sectorIdx] - 0.1,
                                      seedCoords[s_sectorIdx] + 0.1);
      // A middle-layer seed already constrains the track direction more
      // tightly, as the line must connect to hits on both sides. Inner- and
      // outer-layer seeds need a window of double size to achieve the same
      // angular acceptance.
      const double thetaHalfWindow =
          (seedLayer == MuonLayerIndex::Inner ||
           seedLayer == MuonLayerIndex::Outer)
              ? m_cfg.thetaSearchWindow
              : 0.5 * m_cfg.thetaSearchWindow;
      selectRange[s_thetaIdx].shrink(seedCoords[s_thetaIdx] - thetaHalfWindow,
                                     seedCoords[s_thetaIdx] + thetaHalfWindow);

      candidateHits.clear();
      orderedSpacepoints.rangeSearchMapDiscard(
          selectRange, [&candidateHits](const SearchTree_t::coordinate_t&,
                                        const HitPayload* hit) {
            candidateHits.push_back(CandidateHit{hit, hit->station, 0u});
          });
      if (candidateHits.size() < minLayers) {
        ACTS_VERBOSE("findPatternsInEta() Found "
                     << candidateHits.size()
                     << " candidate hits, below the minimum - skip seed.");
        continue;
      }
      // The candidate hits must extend over at least two station layers
      if (std::ranges::none_of(candidateHits,
                               [seedLayer](const CandidateHit& c) {
                                 return layerIndex(c.station) != seedLayer;
                               })) {
        ACTS_VERBOSE("findPatternsInEta() All candidates in the same station "
                     "layer - skip seed.");
        continue;
      }
      // Sort the candidates by global logical layer
      std::ranges::sort(candidateHits, [](const CandidateHit& c1,
                                          const CandidateHit& c2) {
        const LayerOrdering ordering = checkLayerOrdering(*c1, *c2);
        if (ordering == eSameLayer) {
          // Hits on the same layer are sorted by the local precision coordinate
          return c1.sp()->localPosition().y() < c2.sp()->localPosition().y();
        }
        return ordering == eLowerLayer;
      });
      // Assign the global layer number to avoid recomputing it later
      for (std::size_t i = 1u; i < candidateHits.size(); ++i) {
        const bool newLayer = checkLayerOrdering(*candidateHits[i - 1],
                                                 *candidateHits[i]) !=
                              eSameLayer;
        candidateHits[i].globLayer = static_cast<std::uint8_t>(
            candidateHits[i - 1].globLayer + (newLayer ? 1u : 0u));
      }
      if (candidateHits.back().globLayer + 1u < minLayers) {
        ACTS_VERBOSE("findPatternsInEta() Found "
                     << candidateHits.size() << " candidate hits on "
                     << candidateHits.back().globLayer + 1u
                     << " layers, below the minimum - skip seed.");
        continue;
      }
      if (logger().doPrint(Acts::Logging::VERBOSE)) {
        ACTS_VERBOSE("findPatternsInEta() Found " << candidateHits.size()
                                                  << " candidate hits: ");
        for (const CandidateHit& c : candidateHits) {
          ACTS_VERBOSE("findPatternsInEta() \t**" << c);
        }
      }

      // Start the pattern building from the seed
      const auto seedItr = std::ranges::find_if(
          candidateHits, [&seed](const CandidateHit& c) { return c == seed; });
      if (seedItr == candidateHits.end()) {
        ACTS_ERROR("findPatternsInEta() The seed is not among its candidates");
        continue;
      }
      const CandidateHit& seedCand = *seedItr;
      PatternState patternSeed{seedCand,
                               static_cast<std::int8_t>(seedCoords[s_sectorIdx]),
                               &m_cfg, m_logger.get()};

      /// @brief Extend a pattern with a range of hits. Patterns are branched
      ///        when compatible with multiple hits on the same layer. For each
      ///        new hit, extendPatterns tries to extend every active pattern
      ///        and removes the ones not meeting the continuation criteria.
      /// @param begin: Iterator to the first hit to process
      /// @param end: Iterator to the end of the hit range
      /// @param toExtend: Pattern to extend
      auto processHitRange = [&](const auto begin, const auto end,
                                 PatternState&& toExtend) -> PatternStateVec {
        startPatternBuff.clear();
        startPatternBuff.push_back(std::move(toExtend));
        for (auto testItr = begin; testItr != end; ++testItr) {
          const CandidateHit& testHit = *testItr;
          if (testHit.globLayer == seedCand.globLayer) {
            continue;  // skip hits on the same layer as the seed
          }
          extendPatterns(startPatternBuff, endPatternBuff, testHit, beamSpot);
          std::swap(startPatternBuff, endPatternBuff);
        }
        if (startPatternBuff.size() > 1u) {
          return resolveOverlaps(startPatternBuff);
        }
        PatternStateVec extended{};
        if (!startPatternBuff.empty()) {
          extended.push_back(std::move(startPatternBuff.back()));
        }
        return extended;
      };

      // First search for compatible hits from the seed layer outwards
      PatternStateVec forwardExtended = processHitRange(
          std::next(seedItr), candidateHits.end(), std::move(patternSeed));
      ACTS_VERBOSE("findPatternsInEta() Finished forward search, found "
                   << forwardExtended.size()
                   << " patterns, start backward search.");

      PatternStateVec backwardExtended{};
      backwardExtended.reserve(2u * forwardExtended.size());
      for (PatternState& pat : forwardExtended) {
        ACTS_VERBOSE("findPatternsInEta() Start backward search for pattern "
                     << detailed(pat));
        // When inverting the search direction, update the last inserted hit
        // and the line anchor
        pat.moveLineAnchorHit(seedCand);
        pat.lastInsertedHit = seedCand;
        std::ranges::move(processHitRange(std::reverse_iterator{seedItr},
                                          candidateHits.rend(), std::move(pat)),
                          std::back_inserter(backwardExtended));
      }
      if (backwardExtended.size() > 1u) {
        backwardExtended = resolveOverlaps(backwardExtended);
      }

      for (PatternState& pat : backwardExtended) {
        pat.meanNormResidual2 /= pat.nBendingLayers();
        if (!passPatternCuts(pat)) {
          continue;
        }
        ACTS_VERBOSE("findPatternsInEta() Add new pattern " << detailed(pat));
        pat.isFinalized = true;
        outPatterns.push_back(std::move(pat));
      }
    }
  }
  ACTS_VERBOSE("findPatternsInEta() Found in total "
               << outPatterns.size()
               << " patterns in eta before overlap removal");
  return resolveOverlaps(outPatterns);
}

void MuonGlobalPatternFinder::extendPatterns(PatternStateVec& startPatterns,
                                             PatternStateVec& endPatterns,
                                             const CandidateHit& testHit,
                                             const Acts::Vector3& beamSpot) const {
  endPatterns.clear();
  ACTS_VERBOSE("extendPatterns() *** Test " << testHit << " against "
                                            << startPatterns.size()
                                            << " active patterns.");

  // Number of layers between the last inserted hit and the test hit
  auto missedLayers = [&testHit](const PatternState& pat) {
    return static_cast<unsigned>(
        std::abs(pat.lastInsertedHit.globLayer - testHit.globLayer));
  };
  // The minimum number of missed layers among the active patterns is the
  // reference to prune patterns with too many missed layers
  unsigned minMissedLayers{std::numeric_limits<unsigned>::max()};
  for (const PatternState& pat : startPatterns) {
    minMissedLayers = std::min(minMissedLayers, missedLayers(pat));
  }
  const bool shouldPrune =
      startPatterns.size() > 1u &&
      std::ranges::any_of(startPatterns, [](const PatternState& p) {
        return p.nBendingLayers() > 2u;
      });

  for (std::size_t i = 0u; i < startPatterns.size(); ++i) {
    PatternState& pat = startPatterns[i];
    if (pat.isOverlap) {
      continue;
    }
    // Check the pattern has not already missed too many layers compared to
    // the other patterns
    if (pat.lastInsertedHit.station == testHit.station &&
        missedLayers(pat) >
            std::max(m_cfg.maxMissLayersInStation, minMissedLayers)) {
      ACTS_VERBOSE("extendPatterns() Pattern "
                   << detailed(pat) << "\nhas missed " << missedLayers(pat)
                   << " layer hits, above the max allowed - abort pattern.");
      continue;
    }
    // Prune the pattern hypotheses sharing the same last hit, keeping only the
    // best one. All patterns are kept if their last hit is on the test hit
    // layer, to allow further branching.
    if (shouldPrune && pat.lastInsertedHit.globLayer != testHit.globLayer) {
      bool hasBetter{false};
      for (std::size_t j = i + 1u; j < startPatterns.size(); ++j) {
        PatternState& other = startPatterns[j];
        if (other.isOverlap || other.lastInsertedHit != pat.lastInsertedHit) {
          continue;
        }
        if (isBetter(pat, other)) {
          ACTS_VERBOSE("extendPatterns() Pruning: "
                       << detailed(pat) << "\nis BETTER than "
                       << detailed(other));
          other.isOverlap = true;
          continue;
        }
        ACTS_VERBOSE("extendPatterns() Pruning: " << detailed(other)
                                                  << "\nis BETTER than "
                                                  << detailed(pat));
        hasBetter = true;
        break;
      }
      if (hasBetter) {
        continue;
      }
    }
    // Check the line compatibility of the test hit with the pattern
    const LineTestRes res = pat.checkLineComp(testHit, beamSpot);
    switch (res.result) {
      case LineTestDecision::eAddHit: {
        /// @todo Study the feasibility of loosening the criteria for
        ///       low-confidence hits
        const bool lowConfidenceRes =
            res.sigma > m_cfg.lowConfidenceResSigma &&
            res.residual / res.sigma > 2.;
        if (lowConfidenceRes) {
          ACTS_VERBOSE("extendPatterns() Low-confidence hit: residual pull "
                       << res.residual / res.sigma);
          // Keep both the patterns with and without the hit, such that the
          // hit can still be rejected in the next iterations. Make sure first
          // that the forked pattern is original.
          if (std::ranges::any_of(
                  endPatterns, [&testHit, &pat](const PatternState& p) {
                    return p.lastInsertedHit == testHit &&
                           (p.prevLayerHit == pat.lastInsertedHit ||
                            p.nBendingLayers() > pat.nBendingLayers() + 1u);
                  })) {
            ACTS_VERBOSE(
                "extendPatterns() Forking leads to an existing pattern.");
            break;
          }
          endPatterns.push_back(pat);
          endPatterns.back().addHit(testHit, res.residual, res.sigma);
          break;
        }
        ACTS_VERBOSE("extendPatterns() Hit compatible - add to pattern. "
                     "Residual pull "
                     << res.residual / res.sigma);
        pat.addHit(testHit, res.residual, res.sigma);
        break;
      }
      case LineTestDecision::eBranchPattern: {
        // Check first whether the branched pattern already exists
        if (std::ranges::any_of(endPatterns,
                                [&testHit, &pat](const PatternState& p) {
                                  return p.lastInsertedHit == testHit &&
                                         p.prevLayerHit == pat.prevLayerHit;
                                })) {
          ACTS_VERBOSE("extendPatterns() Branched pattern already exists.");
          break;
        }
        // Branch the pattern: clone it and replace its last hit
        ACTS_VERBOSE("extendPatterns() Hit compatible & on the same layer of "
                     "the last added hit - branch pattern.");
        endPatterns.push_back(pat);
        endPatterns.back().overWriteHit(testHit, res.residual, res.sigma);
        break;
      }
      case LineTestDecision::eRejectHit: {
        ACTS_VERBOSE("extendPatterns() Hit is not compatible - reject hit.");
        break;
      }
    }
    endPatterns.push_back(std::move(pat));
  }
  startPatterns.clear();
}

MuonGlobalPatternFinder::PatternStateVec
MuonGlobalPatternFinder::resolveOverlaps(PatternStateVec& toResolve) const {
  ACTS_VERBOSE("resolveOverlaps() Resolving overlaps among "
               << toResolve.size() << " patterns.");
  PatternStateVec outputPatterns{};
  outputPatterns.reserve(toResolve.size());

  /// @brief Check whether two patterns overlap
  auto areOverlapping = [this](const PatternState& a, const PatternState& b) {
    // Check first the geometrical overlap
    if (!a.expSect.isNeighbour(b.expSect) ||
        std::abs(a.patTheta - b.patTheta) > 2. * m_cfg.thetaSearchWindow) {
      return false;
    }
    /// @brief Whether the phi of a pattern is inside both sectors of another
    auto insideSectors = [](const PatternState& withPhi,
                            const PatternState& withoutPhi) {
      return MuonSectorMapping::insideSector(
                 static_cast<int>(withoutPhi.expSect.msSector()),
                 withPhi.patPhi) &&
             MuonSectorMapping::insideSector(
                 static_cast<int>(withoutPhi.expSect.adjacentMsSector()),
                 withPhi.patPhi);
    };
    if (a.nPhiLayers > 0u && b.nPhiLayers > 0u) {
      if (std::abs(Acts::detail::radian_sym(a.patPhi - b.patPhi)) >
          5._degree) {
        return false;
      }
    } else if (a.nPhiLayers > 0u) {
      if (!insideSectors(a, b)) {
        return false;
      }
    } else if (b.nPhiLayers > 0u) {
      if (!insideSectors(b, a)) {
        return false;
      }
    }
    // The patterns can overlap geometrically, so check the hit content
    std::size_t nSharedHits{0u};
    std::size_t nSharedStations{0u};
    for (std::size_t st = 0u; st < s_nStations; ++st) {
      const std::vector<CandidateHit>& hitsA = a.hitsPerStation[st];
      const std::vector<CandidateHit>& hitsB = b.hitsPerStation[st];
      if (hitsA.empty() || hitsB.empty()) {
        continue;
      }
      const auto nSharedInStation = static_cast<std::size_t>(
          std::ranges::count_if(hitsA, [&hitsB](const CandidateHit& hitA) {
            return std::ranges::find(hitsB, hitA) != hitsB.end();
          }));
      nSharedHits += nSharedInStation;
      if (nSharedInStation >= m_cfg.minStationLayers) {
        ++nSharedStations;
      }
    }
    // Overlap if at least 50% of the hits of the smaller pattern are shared
    const std::size_t minHits =
        std::min(a.nBendingLayers(), b.nBendingLayers());
    const std::size_t minStations =
        std::min(a.nStations(true), b.nStations(true));
    return 2u * nSharedHits >= minHits &&
           nSharedStations >= std::min<std::size_t>(2u, minStations);
  };
  /// @brief Determine the best pattern, preferring more good stations
  auto isBetterOverlap = [](const PatternState& a, const PatternState& b) {
    const int nGoodStationDiff = a.nStations(true) - b.nStations(true);
    if (nGoodStationDiff != 0) {
      return nGoodStationDiff > 0;
    }
    return isBetter(a, b);
  };

  for (auto it = toResolve.begin(); it != toResolve.end(); ++it) {
    if (it->isOverlap) {
      continue;
    }
    for (auto jt = std::next(it); jt != toResolve.end(); ++jt) {
      if (jt->isOverlap || !areOverlapping(*it, *jt)) {
        continue;
      }
      if (isBetterOverlap(*it, *jt)) {
        ACTS_VERBOSE("resolveOverlaps() Pattern " << brief(*it)
                                                  << "\nis BETTER than "
                                                  << brief(*jt));
        jt->isOverlap = true;
      } else {
        ACTS_VERBOSE("resolveOverlaps() Pattern " << brief(*jt)
                                                  << "\nis BETTER than "
                                                  << brief(*it));
        it->isOverlap = true;
        break;
      }
    }
    if (!it->isOverlap) {
      outputPatterns.push_back(std::move(*it));
    }
  }
  ACTS_VERBOSE("resolveOverlaps() Patterns surviving overlap removal: "
               << outputPatterns.size());
  return outputPatterns;
}

bool MuonGlobalPatternFinder::passPatternCuts(const PatternState& pat) const {
  const auto nGoodStations = std::ranges::count_if(
      pat.hitsPerStation, [this](const std::vector<CandidateHit>& hits) {
        return hits.size() >= m_cfg.minStationLayers;
      });
  if (pat.nTriggerLayers < m_cfg.minTriggerLayers ||
      pat.nPrecisionLayers < m_cfg.minPrecisionLayers || nGoodStations < 2) {
    ACTS_VERBOSE("passPatternCuts() Pattern "
                 << detailed(pat)
                 << "\ndoes not meet minimum layer requirements - reject.");
    return false;
  }
  if (pat.meanNormResidual2 > m_cfg.meanNormRes2Cut) {
    ACTS_VERBOSE("passPatternCuts() Pattern "
                 << detailed(pat)
                 << "\ndoes not meet the mean norm residual2 cut - reject.");
    return false;
  }
  return true;
}

bool MuonGlobalPatternFinder::isBetter(const PatternState& a,
                                       const PatternState& b) {
  const double resA = a.getMeanResidual2();
  const double resB = b.getMeanResidual2();
  const double resDiff = std::abs(resA - resB) / std::max(resA, resB);
  const int nLayerDiff = a.nBendingLayers() - b.nBendingLayers();
  const int nPrecLayDiff = a.nPrecisionLayers - b.nPrecisionLayers;

  // For patterns that differ by 1-2 layers, don't sacrifice fit quality unless
  // the extra layers are genuinely comparable. For a >= 3-layer difference, the
  // multiplicity advantage is strong enough to dominate.
  if ((nLayerDiff == 0 && nPrecLayDiff == 0) ||
      (std::abs(nLayerDiff) < 3 && resDiff > 0.1)) {
    return resA < resB;
  }
  if (nLayerDiff == 0) {
    return nPrecLayDiff > 0;
  }
  return nLayerDiff > 0;
}

MuonGlobalPatternFinder::LayerOrdering
MuonGlobalPatternFinder::checkLayerOrdering(const HitPayload& hit1,
                                            const HitPayload& hit2) {
  using enum LayerOrdering;
  auto getLayerOrdering = [](bool isLayer1Lower) {
    return isLayer1Lower ? eLowerLayer : eHigherLayer;
  };
  if (hit1 == hit2) {
    return eSameLayer;
  }
  // Hits in the same chamber are ordered by their layer number
  if (hit1.bucket == hit2.bucket) {
    if (hit1.locLayer == hit2.locLayer) {
      return eSameLayer;
    }
    return getLayerOrdering(hit1.locLayer < hit2.locLayer);
  }
  const MuonStationIndex st1 = hit1.station;
  const MuonStationIndex st2 = hit2.station;
  // Hits in the same station but different chambers, e.g. in the overlap
  // region of two adjacent sectors
  if (st1 == st2) {
    const double delta =
        isBarrel(st1)
            ? perp(hit1.position) - perp(hit2.position)
            : std::abs(hit1.position.z()) - std::abs(hit2.position.z());
    if (std::abs(delta) <= Acts::s_epsilon) {
      return eSameLayer;
    }
    return getLayerOrdering(delta < 0.);
  }
  const MuonLayerIndex layer1 = layerIndex(st1);
  const MuonLayerIndex layer2 = layerIndex(st2);
  if (layer1 == layer2) {
    // Hits in different stations but same station layer. Expected to happen
    // only for the Inner and Middle layers
    if (layer1 == MuonLayerIndex::Middle) {
      // The barrel hit comes first
      return getLayerOrdering(st1 == MuonStationIndex::BM);
    }
    if (layer1 == MuonLayerIndex::Inner) {
      // In the large sectors BI comes first, while in the small sectors EI
      // comes first. Use the global radius instead.
      return getLayerOrdering(perp(hit1.position) < perp(hit2.position));
    }
    throw std::runtime_error(
        std::format("MuonGlobalPatternFinder: unexpected pattern-compatible "
                    "hits in {} and {}",
                    toString(st1), toString(st2)));
  }
  if (layer1 == MuonLayerIndex::Inner || layer2 == MuonLayerIndex::Inner) {
    // A hit in the Inner layer comes first
    return getLayerOrdering(layer1 == MuonLayerIndex::Inner);
  }
  if (layer1 == MuonLayerIndex::Outer || layer2 == MuonLayerIndex::Outer) {
    // A hit in the Outer layer comes last
    return getLayerOrdering(layer2 == MuonLayerIndex::Outer);
  }
  if (layer1 == MuonLayerIndex::BarrelExtended ||
      layer2 == MuonLayerIndex::BarrelExtended) {
    // A BarrelExtended hit comes before a Middle or Extended one
    return getLayerOrdering(layer1 == MuonLayerIndex::BarrelExtended);
  }
  // One hit in the Extended (EE) layer and the other one in the Middle layer:
  // EE comes before EM but after BM
  if (layer1 == MuonLayerIndex::Extended) {
    return getLayerOrdering(st2 == MuonStationIndex::EM);
  }
  return getLayerOrdering(st1 == MuonStationIndex::BM);
}

}  // namespace ActsExamples
