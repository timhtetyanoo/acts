// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

/// @brief Local end-to-end check of the muon global pattern finder. It needs a
///        space point n-tuple and the matching tracking geometry, which are not
///        shipped with the repository, so both paths are taken from the
///        environment and the test is skipped when they are unset or missing:
///
///          ACTS_GPF_NTUPLE    ROOT file holding the MuonSpacePoints tree
///          ACTS_GPF_GEOMETRY  matching tracking geometry json file
///
/// @note This file is a development aid and is not meant to be part of an
///       upstream pull request: a unit test does not read external data files.

#include <boost/test/unit_test.hpp>

#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/Framework/AlgorithmContext.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"
#include "ActsExamples/Io/Root/RootMuonSpacePointReader.hpp"
#include "ActsExamples/TrackFinding/GlobalPatternFinderAlgorithm.hpp"
#include "ActsPlugins/Json/TrackingGeometryJsonConverter.hpp"

#include <cstdlib>
#include <filesystem>
#include <string>

namespace ActsTests {

namespace {

/// @brief Returns the path stored in the environment variable or an empty path
///        if the variable is unset or points to a missing file
std::filesystem::path pathFromEnv(const char* variable) {
  const char* value = std::getenv(variable);
  if (value == nullptr) {
    BOOST_TEST_MESSAGE("Environment variable " << variable << " is not set.");
    return {};
  }
  std::filesystem::path path{value};
  if (!std::filesystem::exists(path)) {
    BOOST_TEST_MESSAGE("File " << path.string() << " from " << variable
                               << " does not exist.");
    return {};
  }
  return path;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(GlobalPatternFinderDataSuite)

BOOST_AUTO_TEST_CASE(pattern_finding_from_ntuple) {
  const std::filesystem::path ntuple{pathFromEnv("ACTS_GPF_NTUPLE")};
  const std::filesystem::path geometry{pathFromEnv("ACTS_GPF_GEOMETRY")};
  if (ntuple.empty() || geometry.empty()) {
    BOOST_TEST_MESSAGE("Skipping the data driven pattern finder test.");
    return;
  }

  const auto gctx = Acts::GeometryContext::dangerouslyDefaultConstruct();

  /// The space points are expressed in the frame of their spectrometer sector.
  /// Their measurement surfaces come from the tracking geometry
  Acts::TrackingGeometryJsonConverter geoConverter{};
  std::shared_ptr<Acts::TrackingGeometry> trackingGeometry{
      geoConverter.fromFile(gctx, geometry)};
  BOOST_REQUIRE(trackingGeometry != nullptr);

  ActsExamples::RootMuonSpacePointReader::Config readerCfg{};
  readerCfg.filePath = ntuple.string();
  readerCfg.treeName = "MuonSpacePoints";
  readerCfg.outputSpacePoints = "MuonSpacePoints";
  ActsExamples::RootMuonSpacePointReader reader{readerCfg, Acts::Logging::INFO};

  ActsExamples::GlobalPatternFinderAlgorithm::Config finderCfg{};
  finderCfg.inSpacePoints = readerCfg.outputSpacePoints;
  finderCfg.outPatterns = "MuonGlobalPatterns";
  finderCfg.trackingGeometry = trackingGeometry;
  ActsExamples::GlobalPatternFinderAlgorithm finder{
      finderCfg, Acts::getDefaultLogger("GlobalPatternFinderAlgorithm",
                                        Acts::Logging::DEBUG)};

  ActsExamples::ReadDataHandle<ActsExamples::MuonGlobalPatternContainer>
      patternHandle{&finder, "TestOutputPatterns"};
  patternHandle.initialize(finderCfg.outPatterns);

  const auto [firstEvent, lastEvent] = reader.availableEvents();
  BOOST_REQUIRE_GT(lastEvent, firstEvent);
  BOOST_TEST_MESSAGE("Reading the events [" << firstEvent << ", " << lastEvent
                                            << ") from " << ntuple.string());

  std::size_t totalPatterns{0};
  std::size_t eventsWithPatterns{0};
  for (std::size_t event = firstEvent; event < lastEvent; ++event) {
    /// Each event starts from an empty store, as the sequencer would do
    ActsExamples::WhiteBoard eventStore{};
    ActsExamples::AlgorithmContext ctx{0, event, eventStore, 0};

    BOOST_REQUIRE(reader.read(ctx) == ActsExamples::ProcessCode::SUCCESS);
    BOOST_REQUIRE(finder.execute(ctx) == ActsExamples::ProcessCode::SUCCESS);

    const ActsExamples::MuonGlobalPatternContainer& patterns{
        patternHandle(ctx)};
    totalPatterns += patterns.size();
    eventsWithPatterns += (patterns.empty() ? 0u : 1u);

    BOOST_TEST_MESSAGE("Event " << event << ": found " << patterns.size()
                                << " global patterns.");
    for (const ActsExamples::MuonGlobalPattern& pattern : patterns) {
      std::size_t nHits{0};
      for (const auto& hits : pattern.hitsPerStation) {
        nHits += hits.size();
      }
      BOOST_TEST_MESSAGE(
          "  Pattern in sector "
          << static_cast<int>(pattern.sector) << ", theta: " << pattern.theta
          << ", phi: " << pattern.phi << ", hits: " << nHits
          << ", precision/trigger/phi layers: " << pattern.nPrecisionLayers
          << "/" << pattern.nTriggerLayers << "/" << pattern.nPhiLayers);
      /// Every pattern has to satisfy the cuts it was selected with
      BOOST_CHECK_GE(pattern.nPrecisionLayers, finderCfg.minPrecisionLayers);
      BOOST_CHECK_GT(nHits, 0u);
    }
  }

  BOOST_TEST_MESSAGE("Found " << totalPatterns << " patterns in "
                              << eventsWithPatterns << " of "
                              << (lastEvent - firstEvent) << " events.");
  BOOST_CHECK_GT(totalPatterns, 0u);
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
