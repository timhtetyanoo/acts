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

namespace {

/// @brief Barrel chamber frame: the local x-axis points along phi, the local
///        y-axis along the beam axis & the local z-axis radially outwards
Acts::Transform3 barrelChamberFrame() {
  Acts::Transform3 trf{Acts::Transform3::Identity()};
  trf.linear().col(0) = Acts::Vector3::UnitY();
  trf.linear().col(1) = Acts::Vector3::UnitZ();
  trf.linear().col(2) = Acts::Vector3::UnitX();
  return trf;
}

constexpr double mdtCovEta = 0.01;
constexpr double rpcCovEta = 1.;

/// @brief Hit in the xz-plane of sector 1 expressed in the barrel chamber frame
MuonSpacePoint barrelHit(TechField tech, StationName st, double r, double z,
                         double covEta) {
  MuonSpacePoint sp =
      makeSp(tech, st, true, false, Acts::Vector3{0., z, r},
             Acts::Vector3::UnitX(), Acts::Vector3::UnitY());
  sp.setCovariance(4., covEta, 0.);
  return sp;
}

/// @brief Straight muon from the origin in the xz-plane of sector 1 with
///        z = slope * r. The hits are displaced by +-50 um along z.
void addMuon(MuonSpacePointContainer& container, double slope) {
  const auto addBucket = [&](StationName st, TechField tech,
                             const std::vector<double>& radii, double covEta) {
    MuonSpacePointBucket bucket{};
    for (std::size_t i = 0u; i < radii.size(); ++i) {
      const double offset = (i % 2u == 0u ? 1. : -1.) * 50._um;
      bucket.push_back(barrelHit(tech, st, radii[i],
                                 slope * radii[i] + offset, covEta));
    }
    container.push_back(std::move(bucket));
  };
  addBucket(StationName::BIL, TechField::Mdt, {5._m, 5.026_m, 5.052_m, 5.078_m},
            mdtCovEta);
  addBucket(StationName::BML, TechField::Mdt, {7._m, 7.026_m, 7.052_m, 7.078_m},
            mdtCovEta);
  addBucket(StationName::BML, TechField::Rpc, {7.3_m, 7.31_m}, rpcCovEta);
  addBucket(StationName::BOL, TechField::Mdt, {9._m, 9.026_m, 9.052_m, 9.078_m},
            mdtCovEta);
  addBucket(StationName::BOL, TechField::Rpc, {9.3_m}, rpcCovEta);
}

std::size_t nStationHits(const PatternState& pat, MuonStationIndex st) {
  return pat.hitsPerStation[stationIdx(st)].size();
}

}  // namespace

BOOST_AUTO_TEST_CASE(EtaPatternSingleMuon) {
  constexpr double slope = 0.4;
  MuonSpacePointContainer container{};
  addMuon(container, slope);
  /// Noise hit close to the muon on the second BO MDT layer & a distant noise
  /// hit on the third BM MDT layer
  container[3].push_back(barrelHit(TechField::Mdt, StationName::BOL, 9.026_m,
                                   slope * 9.026_m + 300._um, mdtCovEta));
  container[1].push_back(barrelHit(TechField::Mdt, StationName::BML, 7.052_m,
                                   slope * 7.052_m + 50._mm, mdtCovEta));
  const MuonSpacePoint* closeNoise = &container[3].back();
  const MuonSpacePoint* farNoise = &container[1].back();

  const MuonGlobalPatternFinder finder = makeFinder(barrelChamberFrame());
  const MuonGlobalPatternFinder::SearchTreeData data =
      finder.constructTree(gctx, container);
  const auto patterns = finder.findPatternsInEta(data.tree);

  /// The seeds of all expanded sectors lead to the same pattern
  BOOST_REQUIRE_EQUAL(patterns.size(), 1u);
  const PatternState& pat = patterns.front();
  BOOST_CHECK(pat.isFinalized);
  BOOST_CHECK_EQUAL(static_cast<int>(pat.nPrecisionLayers), 12);
  BOOST_CHECK_EQUAL(static_cast<int>(pat.nTriggerLayers), 3);
  BOOST_CHECK_EQUAL(nStationHits(pat, MuonStationIndex::BI), 4u);
  BOOST_CHECK_EQUAL(nStationHits(pat, MuonStationIndex::BM), 6u);
  BOOST_CHECK_EQUAL(nStationHits(pat, MuonStationIndex::BO), 5u);
  BOOST_CHECK_LE(pat.meanNormResidual2, finder.config().meanNormRes2Cut);
  CHECK_CLOSE_ABS(pat.patTheta, std::atan2(1., slope), 1e-3);

  /// The noise hits are not part of the pattern
  for (const HitPayload& hit : data.hitPayloads) {
    const bool isNoise = hit.sp == closeNoise || hit.sp == farNoise;
    BOOST_CHECK_EQUAL(pat.isInPattern(hit), !isNoise);
  }
}

BOOST_AUTO_TEST_CASE(EtaPatternTwoMuons) {
  MuonSpacePointContainer container{};
  addMuon(container, 0.4);
  addMuon(container, 0.2);

  const MuonGlobalPatternFinder finder = makeFinder(barrelChamberFrame());
  const MuonGlobalPatternFinder::SearchTreeData data =
      finder.constructTree(gctx, container);
  auto patterns = finder.findPatternsInEta(data.tree);

  BOOST_REQUIRE_EQUAL(patterns.size(), 2u);
  std::ranges::sort(patterns, {}, &PatternState::patTheta);
  CHECK_CLOSE_ABS(patterns[0].patTheta, std::atan2(1., 0.4), 1e-3);
  CHECK_CLOSE_ABS(patterns[1].patTheta, std::atan2(1., 0.2), 1e-3);
  for (const PatternState& pat : patterns) {
    BOOST_CHECK_EQUAL(static_cast<int>(pat.nBendingLayers()), 15);
  }
}

BOOST_AUTO_TEST_CASE(EtaPatternTooFewLayers) {
  MuonSpacePointContainer container{};
  addMuon(container, 0.4);
  /// Keep only the BM chambers
  container.erase(container.begin() + 3, container.end());
  container.erase(container.begin());

  const MuonGlobalPatternFinder finder = makeFinder(barrelChamberFrame());
  const MuonGlobalPatternFinder::SearchTreeData data =
      finder.constructTree(gctx, container);
  BOOST_CHECK(finder.findPatternsInEta(data.tree).empty());
}

BOOST_AUTO_TEST_CASE(OverlapRemoval) {
  using PatternStateVec = MuonGlobalPatternFinder::PatternStateVec;
  const MuonGlobalPatternFinder finder = makeFinder();
  const auto& cfg = finder.config();
  MuonSpacePointBucket biBucket{};
  MuonSpacePointBucket bmBucket{};
  MuonSpacePointBucket bmOtherBucket{};
  MuonSpacePointBucket boBucket{};
  for (unsigned i = 0u; i < 4u; ++i) {
    biBucket.push_back(makeEta(StationName::BIL, {5._m, 0., 1._m + i * 1._cm}));
    bmBucket.push_back(makeEta(StationName::BML, {7._m, 0., 2._m + i * 1._cm}));
    bmOtherBucket.push_back(
        makeEta(StationName::BML, {7._m, 0., 3._m + i * 1._cm}));
    boBucket.push_back(makeEta(StationName::BOL, {9._m, 0., 3._m + i * 1._cm}));
  }
  std::vector<HitPayload> payloads{};
  payloads.reserve(16u);
  for (const MuonSpacePointBucket* bucket :
       {&biBucket, &bmBucket, &bmOtherBucket, &boBucket}) {
    for (std::size_t i = 0u; i < bucket->size(); ++i) {
      payloads.push_back(makePayload(*bucket, i));
    }
  }
  const auto stationHits = [&payloads](std::size_t first) {
    std::vector<CandidateHit> hits{};
    for (std::size_t i = first; i < first + 4u; ++i) {
      hits.push_back(CandidateHit{&payloads[i], payloads[i].station, 0u});
    }
    return hits;
  };
  /// Pattern with 4 hits in BI and 4 hits in either of the BM chambers
  const auto pattern = [&](double res, bool otherBm = false) {
    PatternState pat = makePattern(finder, payloads[0], 8u, 0u, res);
    pat.hitsPerStation[stationIdx(MuonStationIndex::BI)] = stationHits(0u);
    pat.hitsPerStation[stationIdx(MuonStationIndex::BM)] =
        stationHits(otherBm ? 8u : 4u);
    return pat;
  };
  const auto resolve = [&finder](PatternStateVec patterns) {
    return finder.resolveOverlaps(patterns);
  };

  /// Identical hit content: the better residual wins
  PatternStateVec sameHits{};
  sameHits.push_back(pattern(2.));
  sameHits.push_back(pattern(1.));
  const PatternStateVec resolved = resolve(std::move(sameHits));
  BOOST_REQUIRE_EQUAL(resolved.size(), 1u);
  CHECK_CLOSE_REL(resolved.front().meanNormResidual2, 1., 1e-12);

  /// Patterns far apart in theta or sector do not overlap
  PatternStateVec thetaApart{};
  thetaApart.push_back(pattern(1.));
  thetaApart.push_back(pattern(1.));
  thetaApart.back().patTheta += 3. * cfg.thetaSearchWindow;
  BOOST_CHECK_EQUAL(resolve(std::move(thetaApart)).size(), 2u);

  PatternStateVec sectorApart{};
  sectorApart.push_back(pattern(1.));
  sectorApart.push_back(pattern(1.));
  sectorApart.back().expSect =
      MuonExpandedSector{5u, SectorProjector::center};
  BOOST_CHECK_EQUAL(resolve(std::move(sectorApart)).size(), 2u);

  /// Patterns with phi measurements are compared in phi
  const auto phiPatterns = [&](double phi1, double phi2) {
    PatternStateVec patterns{};
    for (const double patPhi : {phi1, phi2}) {
      patterns.push_back(pattern(1.));
      patterns.back().nPhiLayers = 1u;
      patterns.back().patPhi = patPhi;
    }
    return patterns;
  };
  BOOST_CHECK_EQUAL(resolve(phiPatterns(0., 10._degree)).size(), 2u);
  BOOST_CHECK_EQUAL(resolve(phiPatterns(0., 2._degree)).size(), 1u);
  /// A pattern phi outside the sector of the pattern without phi
  PatternStateVec outsideSector{};
  outsideSector.push_back(pattern(1.));
  outsideSector.push_back(pattern(1.));
  outsideSector.back().nPhiLayers = 1u;
  outsideSector.back().patPhi = 1.;
  BOOST_CHECK_EQUAL(resolve(std::move(outsideSector)).size(), 2u);

  /// More good stations win over a better residual
  PatternStateVec moreStations{};
  moreStations.push_back(pattern(1.));
  moreStations.push_back(pattern(3.));
  moreStations.back().nPrecisionLayers = 12u;
  moreStations.back().hitsPerStation[stationIdx(MuonStationIndex::BO)] =
      stationHits(12u);
  const PatternStateVec resolvedStations = resolve(std::move(moreStations));
  BOOST_REQUIRE_EQUAL(resolvedStations.size(), 1u);
  BOOST_CHECK_EQUAL(static_cast<int>(resolvedStations.front().nStations(true)),
                    3);

  /// Sharing only one good station is not an overlap
  PatternStateVec oneSharedStation{};
  oneSharedStation.push_back(pattern(1.));
  oneSharedStation.push_back(pattern(1., true));
  BOOST_CHECK_EQUAL(resolve(std::move(oneSharedStation)).size(), 2u);
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
