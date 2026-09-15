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
#include "Acts/Utilities/Helpers.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/EventData/MuonGlobalPatternFinderEvent.hpp"
#include "ActsExamples/EventData/MuonSpacePoint.hpp"
#include "ActsExamples/TrackFinding/MuonGlobalPatternFinder.hpp"
#include "ActsExamples/TrackFinding/MuonGlobalPatternFinderDefs.hpp"
#include "ActsTests/CommonHelpers/FloatComparisons.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

using namespace ActsExamples;
using namespace ActsExamples::MuonGlobalPatternFinderDefs;
using namespace Acts::UnitLiterals;

namespace ActsTests {

namespace {

using MuonId = MuonSpacePoint::MuonId;
using StationName = MuonId::StationName;
using TechField = MuonId::TechField;
using SectorProjector = MuonExpandedSector::SectorProjector;
using LayerOrdering = MuonGlobalPatternFinder::LayerOrdering;
using SeedCoords = MuonGlobalPatternFinder::SeedCoords;

const auto gctx = Acts::GeometryContext::dangerouslyDefaultConstruct();
const Acts::Transform3 identity{Acts::Transform3::Identity()};
const auto testLogger =
    Acts::getDefaultLogger("MuonGlobalPatternFinderTests", Acts::Logging::INFO);

MuonSpacePoint makeSp(TechField tech, StationName st, bool measEta,
                      bool measPhi, const Acts::Vector3& pos,
                      const Acts::Vector3& sensorDir = Acts::Vector3::UnitY(),
                      const Acts::Vector3& toNext = Acts::Vector3::UnitZ(),
                      std::uint16_t sector = 1u) {
  MuonId id{};
  id.setChamber(st, MuonId::DetSide::A, sector, tech);
  id.setCoordFlags(measEta, measPhi);
  MuonSpacePoint sp{};
  sp.setId(id);
  sp.defineCoordinates(Acts::Vector3{pos}, Acts::Vector3{sensorDir},
                       Acts::Vector3{toNext});
  sp.setCovariance(4., 9., 0.);
  return sp;
}

MuonSpacePoint makeEta(StationName st, const Acts::Vector3& pos) {
  return makeSp(TechField::Mdt, st, true, false, pos);
}

HitPayload makePayload(const MuonSpacePointBucket& bucket, std::size_t idx,
                       std::uint8_t locLayer = 0u) {
  const MuonSpacePoint& sp = bucket.at(idx);
  return HitPayload{&sp, &bucket, identity, locLayer,
                    stationIndex(sp.id().msStation())};
}

MuonGlobalPatternFinder makeFinder(
    const Acts::Transform3& localToGlobal = identity, bool useMdtHits = true) {
  MuonGlobalPatternFinder::Config cfg{};
  cfg.useMdtHits = useMdtHits;
  cfg.localToGlobal = [localToGlobal](const Acts::GeometryContext&,
                                      const MuonSpacePointBucket&) {
    return localToGlobal;
  };
  return MuonGlobalPatternFinder{
      std::move(cfg),
      Acts::getDefaultLogger("MuonGlobalPatternFinder", Acts::Logging::INFO)};
}

/// @brief Check the ordering of two hits and its mirrored counterpart
void checkOrdering(const HitPayload& hit1, const HitPayload& hit2,
                   LayerOrdering expected) {
  const LayerOrdering mirrored = expected == LayerOrdering::eLowerLayer
                                     ? LayerOrdering::eHigherLayer
                                 : expected == LayerOrdering::eHigherLayer
                                     ? LayerOrdering::eLowerLayer
                                     : LayerOrdering::eSameLayer;
  BOOST_CHECK(MuonGlobalPatternFinder::checkLayerOrdering(hit1, hit2) ==
              expected);
  BOOST_CHECK(MuonGlobalPatternFinder::checkLayerOrdering(hit2, hit1) ==
              mirrored);
}

/// @brief Pattern with the given counts and final mean residual
PatternState makePattern(const MuonGlobalPatternFinder& finder,
                         const HitPayload& seed, unsigned nPrec, unsigned nTrig,
                         double meanRes2) {
  PatternState pat{CandidateHit{&seed, seed.station, 0u},
                   MuonExpandedSector{1u, SectorProjector::center}.sector(),
                   &finder.config(), testLogger.get()};
  pat.nPrecisionLayers = static_cast<std::uint8_t>(nPrec);
  pat.nTriggerLayers = static_cast<std::uint8_t>(nTrig);
  pat.meanNormResidual2 = meanRes2;
  pat.isFinalized = true;
  return pat;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(MuonGlobalPatternFinderSuite)

BOOST_AUTO_TEST_CASE(Construction) {
  MuonGlobalPatternFinder::Config cfg{};
  BOOST_CHECK_THROW(
      MuonGlobalPatternFinder(cfg, Acts::getDefaultLogger(
                                       "Finder", Acts::Logging::INFO)),
      std::invalid_argument);
  const MuonGlobalPatternFinder finder = makeFinder();
  BOOST_CHECK(finder.config().useMdtHits);
  BOOST_CHECK_EQUAL(finder.config().minPrecisionLayers, 8u);
}

BOOST_AUTO_TEST_CASE(LayerOrderingTest) {
  using enum LayerOrdering;
  /// Barrel chambers
  const MuonSpacePointBucket bil{makeEta(StationName::BIL, {5._m, 0., 1._m}),
                                 makeEta(StationName::BIL, {5.03_m, 0., 1._m})};
  const MuonSpacePointBucket bilOverlap{
      makeEta(StationName::BIL, {5.1_m, 0., 1._m}),
      makeEta(StationName::BIL, {5._m, 0., 1.2_m})};
  const MuonSpacePointBucket bml{makeEta(StationName::BML, {7._m, 0., 2._m})};
  const MuonSpacePointBucket bol{makeEta(StationName::BOL, {9._m, 0., 3._m})};
  const MuonSpacePointBucket bee{makeEta(StationName::BEE, {4.5_m, 0., 5._m})};
  /// Endcap chambers
  const MuonSpacePointBucket eil{makeEta(StationName::EIL, {2._m, 0., 7.4_m})};
  const MuonSpacePointBucket eilOverlap{
      makeEta(StationName::EIL, {2._m, 0., 7.5_m})};
  const MuonSpacePointBucket eel{makeEta(StationName::EEL, {4._m, 0., 10._m})};
  const MuonSpacePointBucket eml{makeEta(StationName::EML, {5._m, 0., 14._m})};
  const MuonSpacePointBucket eol{makeEta(StationName::EOL, {6._m, 0., 21._m})};

  const HitPayload bil0 = makePayload(bil, 0u, 0u);
  const HitPayload bil1 = makePayload(bil, 1u, 1u);
  const HitPayload bil1OnLayer0 = makePayload(bil, 1u, 0u);
  const HitPayload bilOther = makePayload(bilOverlap, 0u);
  const HitPayload bilOtherSameR = makePayload(bilOverlap, 1u);
  const HitPayload bm = makePayload(bml, 0u);
  const HitPayload bo = makePayload(bol, 0u);
  const HitPayload be = makePayload(bee, 0u);
  const HitPayload ei = makePayload(eil, 0u);
  const HitPayload eiOther = makePayload(eilOverlap, 0u);
  const HitPayload ee = makePayload(eel, 0u);
  const HitPayload em = makePayload(eml, 0u);
  const HitPayload eo = makePayload(eol, 0u);

  /// Same hit & same chamber
  checkOrdering(bil0, bil0, eSameLayer);
  checkOrdering(bil0, bil1, eLowerLayer);
  checkOrdering(bil0, bil1OnLayer0, eSameLayer);
  /// Same station, different chambers: radius in the barrel, |z| in the endcap
  checkOrdering(bil0, bilOther, eLowerLayer);
  checkOrdering(bil0, bilOtherSameR, eSameLayer);
  checkOrdering(ei, eiOther, eLowerLayer);
  /// Inner before Middle before Outer
  checkOrdering(bil0, bm, eLowerLayer);
  checkOrdering(bm, bo, eLowerLayer);
  checkOrdering(bil0, bo, eLowerLayer);
  checkOrdering(ei, em, eLowerLayer);
  checkOrdering(em, eo, eLowerLayer);
  checkOrdering(ei, bm, eLowerLayer);
  /// Same station layer: BM before EM, BI vs EI by radius
  checkOrdering(bm, em, eLowerLayer);
  checkOrdering(ei, bil0, eLowerLayer);
  /// BarrelExtended before Middle & Extended, after Inner, before Outer
  checkOrdering(be, bm, eLowerLayer);
  checkOrdering(be, em, eLowerLayer);
  checkOrdering(be, ee, eLowerLayer);
  checkOrdering(bil0, be, eLowerLayer);
  checkOrdering(be, bo, eLowerLayer);
  /// Extended before EM, but after BM
  checkOrdering(ee, em, eLowerLayer);
  checkOrdering(bm, ee, eLowerLayer);
  /// Outer hits in barrel & endcap are not expected in the same pattern
  BOOST_CHECK_THROW(MuonGlobalPatternFinder::checkLayerOrdering(bo, eo),
                    std::runtime_error);
}

BOOST_AUTO_TEST_CASE(PatternComparison) {
  const MuonGlobalPatternFinder finder = makeFinder();
  const MuonSpacePointBucket bucket{
      makeEta(StationName::BIL, {5._m, 0., 1._m})};
  const HitPayload seed = makePayload(bucket, 0u);
  const auto pattern = [&](unsigned nPrec, unsigned nTrig, double res) {
    return makePattern(finder, seed, nPrec, nTrig, res);
  };
  const auto checkBetter = [](const PatternState& a, const PatternState& b) {
    BOOST_CHECK(MuonGlobalPatternFinder::isBetter(a, b));
    BOOST_CHECK(!MuonGlobalPatternFinder::isBetter(b, a));
  };
  /// Same layer content: the residual decides
  checkBetter(pattern(8u, 2u, 1.), pattern(8u, 2u, 2.));
  /// 1-2 more layers with a clearly worse residual: the residual decides
  checkBetter(pattern(8u, 2u, 1.), pattern(9u, 2u, 2.));
  /// 1-2 more layers with a comparable residual: the layers decide
  checkBetter(pattern(9u, 2u, 1.05), pattern(8u, 2u, 1.));
  /// >= 3 more layers: the layers decide regardless of the residual
  checkBetter(pattern(11u, 2u, 3.), pattern(8u, 2u, 1.));
  /// Same bending layers & comparable residual: the precision layers decide
  checkBetter(pattern(9u, 1u, 1.02), pattern(8u, 2u, 1.));
  /// Same bending layers & clearly different residual: the residual decides
  checkBetter(pattern(8u, 2u, 1.), pattern(9u, 1u, 2.));
}

BOOST_AUTO_TEST_CASE(PatternCuts) {
  const MuonGlobalPatternFinder finder = makeFinder();
  const auto& cfg = finder.config();
  const MuonSpacePointBucket biBucket{
      makeEta(StationName::BIL, {5._m, 0., 1._m})};
  const MuonSpacePointBucket bmBucket{
      makeEta(StationName::BML, {7._m, 0., 2._m})};
  const HitPayload bi = makePayload(biBucket, 0u);
  const HitPayload bm = makePayload(bmBucket, 0u);

  /// Pattern with minStationLayers hits in BI and BM
  const auto goodPattern = [&]() {
    PatternState pat = makePattern(finder, bi, cfg.minPrecisionLayers,
                                   cfg.minTriggerLayers, 1.);
    auto& biHits = pat.hitsPerStation[stationIdx(MuonStationIndex::BI)];
    auto& bmHits = pat.hitsPerStation[stationIdx(MuonStationIndex::BM)];
    biHits.assign(cfg.minStationLayers, CandidateHit{&bi, bi.station, 0u});
    bmHits.assign(cfg.minStationLayers, CandidateHit{&bm, bm.station, 0u});
    return pat;
  };
  BOOST_CHECK(finder.passPatternCuts(goodPattern()));

  PatternState fewPrecision = goodPattern();
  --fewPrecision.nPrecisionLayers;
  BOOST_CHECK(!finder.passPatternCuts(fewPrecision));

  PatternState fewTrigger = goodPattern();
  fewTrigger.nTriggerLayers = 0u;
  BOOST_CHECK(!finder.passPatternCuts(fewTrigger));

  PatternState oneGoodStation = goodPattern();
  oneGoodStation.hitsPerStation[stationIdx(MuonStationIndex::BM)].pop_back();
  BOOST_CHECK(!finder.passPatternCuts(oneGoodStation));

  PatternState badResidual = goodPattern();
  badResidual.meanNormResidual2 = 1.01 * cfg.meanNormRes2Cut;
  BOOST_CHECK(!finder.passPatternCuts(badResidual));
}

BOOST_AUTO_TEST_CASE(SearchTree) {
  const double overlapPhi = MuonSectorMapping::sectorOverlapPhi(1, 2);
  const Acts::Vector3 overlapPos{7._m * std::cos(overlapPhi),
                                 7._m * std::sin(overlapPhi), 3.2_m};
  const Acts::Vector3 overlapTangent{-std::sin(overlapPhi),
                                     std::cos(overlapPhi), 0.};

  MuonSpacePointContainer container{};
  /// Chamber in the centre of sector 1 with an MDT, an eta-phi RPC and a
  /// phi-only TGC hit on three different layers
  container.push_back(MuonSpacePointBucket{
      makeEta(StationName::BML, {7._m, 0., 3._m}),
      makeSp(TechField::Rpc, StationName::BML, true, true, {7._m, 0., 3.1_m}),
      makeSp(TechField::Tgc, StationName::BML, false, true, {7._m, 0., 3.2_m},
             Acts::Vector3::UnitZ(), Acts::Vector3::UnitY())});
  container.emplace_back();
  /// Eta-phi RPC hit in the overlap between sector 1 & 2
  container.push_back(MuonSpacePointBucket{
      makeSp(TechField::Rpc, StationName::BML, true, true, overlapPos,
             overlapTangent)});
  const MuonSpacePoint* mdt = &container[0][0];
  const MuonSpacePoint* rpc = &container[0][1];
  const MuonSpacePoint* rpcOverlap = &container[2][0];

  const MuonGlobalPatternFinder finder = makeFinder();
  const MuonGlobalPatternFinder::SearchTreeData data =
      finder.constructTree(gctx, container);

  /// Phi-only hits are not added
  BOOST_REQUIRE_EQUAL(data.hitPayloads.size(), 3u);
  for (const HitPayload& hit : data.hitPayloads) {
    BOOST_CHECK_EQUAL(hit.station, MuonStationIndex::BM);
    BOOST_CHECK(hit.sp != &container[0][2]);
  }
  BOOST_CHECK_EQUAL(static_cast<int>(data.hitPayloads[0].locLayer), 0);
  BOOST_CHECK_EQUAL(static_cast<int>(data.hitPayloads[1].locLayer), 1);
  BOOST_CHECK_EQUAL(data.hitPayloads[1].bucket, &container[0]);
  BOOST_CHECK_EQUAL(data.hitPayloads[2].bucket, &container[2]);

  /// Collect the expanded sectors & theta values per space point
  std::map<const MuonSpacePoint*, std::map<int, double>> entries{};
  for (const auto& [coords, hit] : data.tree) {
    const int sector = static_cast<int>(
        coords[Acts::toUnderlying(SeedCoords::eSector)]);
    entries[hit->sp][sector] = coords[Acts::toUnderlying(SeedCoords::eTheta)];
  }
  BOOST_CHECK_EQUAL(std::distance(data.tree.begin(), data.tree.end()), 6);

  const auto rawSector = [](SectorProjector proj) {
    return static_cast<int>(MuonExpandedSector{1u, proj}.sector());
  };
  /// The MDT hit is added to all expanded sectors of its MS sector. The radius
  /// is projected onto the expanded sector plane
  BOOST_REQUIRE_EQUAL(entries[mdt].size(), 3u);
  for (const auto proj : {SectorProjector::leftOverlap, SectorProjector::center,
                          SectorProjector::rightOverlap}) {
    const double sectorPhi = MuonExpandedSector{1u, proj}.phi();
    BOOST_REQUIRE(entries[mdt].contains(rawSector(proj)));
    CHECK_CLOSE_ABS(entries[mdt][rawSector(proj)],
                    std::atan2(7._m * std::cos(sectorPhi), 3._m), 1e-12);
  }
  /// The eta-phi hit in the sector centre is only added to the centre
  BOOST_REQUIRE_EQUAL(entries[rpc].size(), 1u);
  CHECK_CLOSE_ABS(entries[rpc][rawSector(SectorProjector::center)],
                  std::atan2(7._m, 3.1_m), 1e-12);
  /// The eta-phi hit in the overlap is added to the centre & the overlap
  BOOST_REQUIRE_EQUAL(entries[rpcOverlap].size(), 2u);
  BOOST_CHECK(entries[rpcOverlap].contains(rawSector(SectorProjector::center)));
  BOOST_CHECK(
      entries[rpcOverlap].contains(rawSector(SectorProjector::rightOverlap)));
  CHECK_CLOSE_ABS(entries[rpcOverlap][rawSector(SectorProjector::center)],
                  std::atan2(7._m, 3.2_m), 1e-12);

  /// MDT hits can be excluded
  const MuonGlobalPatternFinder noMdtFinder = makeFinder(identity, false);
  const MuonGlobalPatternFinder::SearchTreeData noMdtData =
      noMdtFinder.constructTree(gctx, container);
  BOOST_CHECK_EQUAL(noMdtData.hitPayloads.size(), 2u);
  BOOST_CHECK_EQUAL(
      std::distance(noMdtData.tree.begin(), noMdtData.tree.end()), 3);

  /// The bucket transform is applied to the hit positions
  const Acts::Transform3 shift{Acts::Translation3{0., 0., 1._m}};
  const MuonGlobalPatternFinder shiftedFinder = makeFinder(shift);
  const MuonGlobalPatternFinder::SearchTreeData shiftedData =
      shiftedFinder.constructTree(gctx, container);
  CHECK_CLOSE_ABS(shiftedData.hitPayloads.front().position,
                  Acts::Vector3(7._m, 0., 4._m), 1e-9);

  /// Empty event
  const MuonGlobalPatternFinder::SearchTreeData emptyData =
      finder.constructTree(gctx, MuonSpacePointContainer{});
  BOOST_CHECK(emptyData.hitPayloads.empty());
  BOOST_CHECK(emptyData.tree.begin() == emptyData.tree.end());
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
