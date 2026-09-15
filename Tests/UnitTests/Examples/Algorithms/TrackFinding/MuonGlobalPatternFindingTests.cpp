// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <boost/test/unit_test.hpp>

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/EventData/MuonGlobalPatternFinderEvent.hpp"
#include "ActsExamples/EventData/MuonSpacePoint.hpp"
#include "ActsExamples/Framework/AlgorithmContext.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/Framework/ProcessCode.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"
#include "ActsExamples/TrackFinding/MuonGlobalPatternFinding.hpp"

#include <algorithm>
#include <stdexcept>

using namespace ActsExamples;
using namespace Acts::UnitLiterals;

namespace ActsTests {

namespace {

using MuonId = MuonSpacePoint::MuonId;

MuonGlobalPatternFinding::Config makeConfig() {
  MuonGlobalPatternFinding::Config cfg{};
  cfg.inputSpacePoints = "MuonSpacePoints";
  cfg.outputPatterns = "MuonGlobalPatterns";
  cfg.localToGlobal = [](const Acts::GeometryContext&,
                         const MuonSpacePointBucket&) {
    return Acts::Transform3{Acts::Transform3::Identity()};
  };
  return cfg;
}

std::unique_ptr<const Acts::Logger> makeLogger() {
  return Acts::getDefaultLogger("MuonGlobalPatternFinding",
                                Acts::Logging::INFO);
}

MuonSpacePoint makeSp(MuonId::StationName st, MuonId::TechField tech,
                      bool measPhi, const Acts::Vector3& pos, double covEta) {
  MuonId id{};
  id.setChamber(st, MuonId::DetSide::A, 1u, tech);
  id.setCoordFlags(true, measPhi);
  MuonSpacePoint sp{};
  sp.setId(id);
  sp.defineCoordinates(Acts::Vector3{pos},
                       Acts::Vector3{Acts::Vector3::UnitY()},
                       Acts::Vector3{Acts::Vector3::UnitZ()});
  sp.setCovariance(4., covEta, 0.);
  return sp;
}

/// @brief Event with an MDT hit in BI and an eta-phi RPC hit in BM on a
///        straight line from the origin, defined in the global frame
MuonSpacePointContainer makeEvent() {
  MuonSpacePointContainer spacePoints{};
  spacePoints.push_back(MuonSpacePointBucket{
      makeSp(MuonId::StationName::BIL, MuonId::TechField::Mdt, false,
             Acts::Vector3{5._m, 0., 2._m}, 0.01)});
  spacePoints.push_back(MuonSpacePointBucket{
      makeSp(MuonId::StationName::BML, MuonId::TechField::Rpc, true,
             Acts::Vector3{7._m, 0., 2.8_m + 1._mm}, 1.)});
  return spacePoints;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(MuonGlobalPatternFindingSuite)

BOOST_AUTO_TEST_CASE(Configuration) {
  MuonGlobalPatternFinding::Config noInput = makeConfig();
  noInput.inputSpacePoints.clear();
  BOOST_CHECK_THROW(MuonGlobalPatternFinding(noInput, makeLogger()),
                    std::invalid_argument);

  MuonGlobalPatternFinding::Config noOutput = makeConfig();
  noOutput.outputPatterns.clear();
  BOOST_CHECK_THROW(MuonGlobalPatternFinding(noOutput, makeLogger()),
                    std::invalid_argument);

  MuonGlobalPatternFinding::Config noTransform = makeConfig();
  noTransform.localToGlobal = nullptr;
  BOOST_CHECK_THROW(MuonGlobalPatternFinding(noTransform, makeLogger()),
                    std::invalid_argument);

  /// The configuration is forwarded to the pattern finder
  MuonGlobalPatternFinding::Config cfg = makeConfig();
  cfg.minPrecisionLayers = 6u;
  const MuonGlobalPatternFinding alg{cfg, makeLogger()};
  const auto& finderCfg = alg.patternFinder().config();
  BOOST_CHECK_EQUAL(finderCfg.minPrecisionLayers, 6u);
  BOOST_CHECK_EQUAL(finderCfg.layerSeedings.size(), 2u);

  /// Seeding from the Inner layer is appended to the default seeding layers
  cfg.seedFromInner = true;
  const MuonGlobalPatternFinding innerAlg{cfg, makeLogger()};
  const auto& seedings = innerAlg.patternFinder().config().layerSeedings;
  BOOST_CHECK_EQUAL(seedings.size(), 3u);
  BOOST_CHECK(std::ranges::find(seedings, MuonLayerIndex::Inner) !=
              seedings.end());
}

BOOST_AUTO_TEST_CASE(Execute) {
  /// Loose cuts such that the two-hit event makes a pattern
  MuonGlobalPatternFinding::Config cfg = makeConfig();
  cfg.minPrecisionLayers = 0u;
  cfg.minTriggerLayers = 1u;
  cfg.minStationLayers = 1u;
  cfg.minPhiLayers = 0u;
  const MuonGlobalPatternFinding alg{cfg, makeLogger()};

  WhiteBoard eventStore{};
  AlgorithmContext ctx{0, 0, eventStore, 0};

  WriteDataHandle<MuonSpacePointContainer> inputHandle{&alg,
                                                       "TestInputSpacePoints"};
  inputHandle.initialize(cfg.inputSpacePoints);
  inputHandle(ctx, makeEvent());

  BOOST_REQUIRE(alg.execute(ctx) == ProcessCode::SUCCESS);

  ReadDataHandle<MuonGlobalPatternContainer> outputHandle{&alg,
                                                          "TestOutputPatterns"};
  outputHandle.initialize(cfg.outputPatterns);
  const MuonGlobalPatternContainer& patterns = outputHandle(ctx);
  BOOST_REQUIRE_EQUAL(patterns.size(), 1u);

  const MuonGlobalPattern& pattern = patterns.front();
  BOOST_CHECK_EQUAL(pattern.nPrecisionLayers(), 1u);
  BOOST_CHECK_EQUAL(pattern.nTriggerLayers(), 1u);
  BOOST_CHECK_EQUAL(pattern.nPhiLayers(), 1u);
  BOOST_CHECK_EQUAL(pattern.sector(), 1u);

  /// The patterns refer to the space points in the event store
  ReadDataHandle<MuonSpacePointContainer> storedHandle{&alg,
                                                       "TestStoredSpacePoints"};
  storedHandle.initialize(cfg.inputSpacePoints);
  const MuonSpacePointContainer& stored = storedHandle(ctx);
  const auto& biHits = pattern.hitsInStation(MuonStationIndex::BI);
  const auto& bmHits = pattern.hitsInStation(MuonStationIndex::BM);
  BOOST_REQUIRE_EQUAL(biHits.size(), 1u);
  BOOST_REQUIRE_EQUAL(bmHits.size(), 1u);
  BOOST_CHECK_EQUAL(biHits.front(), &stored[0][0]);
  BOOST_CHECK_EQUAL(bmHits.front(), &stored[1][0]);
  BOOST_CHECK_EQUAL(pattern.bucketsInStation(MuonStationIndex::BM).front(),
                    &stored[1]);
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
