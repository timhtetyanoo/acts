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
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/Surfaces/RectangleBounds.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/EventData/CudaPatternFinderData.hpp"
#include "ActsExamples/EventData/MuonSpacePoint.hpp"
#include "ActsExamples/TrackFinding/CudaPatternFinder.hpp"
#include "ActsExamples/TrackFinding/GlobalPatternFinderAlgorithm.hpp"
#include "ActsExamples/Utilities/CudaStream.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

using namespace Acts::UnitLiterals;

namespace ActsTests {

using namespace Acts;
namespace Math = ActsExamples::CudaPatternMath;

using MuonId = ActsExamples::MuonSpacePoint::MuonId;
using PatternStateAux = ActsExamples::GlobalPatternFinder_t::PatternStateAux;
using OrderedHit = PatternStateAux::OrderedHit;

namespace {
ActsExamples::MuonSpacePoint makeSpacePoint(MuonId::StationName station,
                                            MuonId::TechField tech,
                                            bool measPhi, const Vector3& pos,
                                            std::uint8_t layer = 2u) {
  MuonId id{};
  id.setChamber(station, MuonId::DetSide::A, 3u, tech);
  id.setLayAndCh(layer, 5u);
  id.setCoordFlags(true, measPhi, false);

  ActsExamples::MuonSpacePoint sp{};
  sp.setId(id);
  sp.setGeometryId(GeometryIdentifier{7u});
  sp.defineCoordinates(Vector3{pos}, Vector3::UnitY(), Vector3::UnitX());
  sp.setCovariance(0.09, 4., 0.);
  sp.setRadius(7.);
  return sp;
}

/// @brief Hits of every station layer, as payloads and as their device copy
struct Hits {
  ActsExamples::MuonSpacePointBucket bucket{};
  std::shared_ptr<Surface> surface{};
  std::vector<ActsExamples::HitPayload> payloads{};
  std::vector<Math::Hit> device{};

  explicit Hits(const GeometryContext& gctx) {
    using SN = MuonId::StationName;
    using TF = MuonId::TechField;
    Transform3 surfaceTrf{Transform3::Identity()};
    surfaceTrf.rotate(AngleAxis3{0.3, Vector3::UnitZ()});
    surfaceTrf.rotate(AngleAxis3{0.2, Vector3::UnitX()});
    surface = Surface::makeShared<PlaneSurface>(
        surfaceTrf, std::make_shared<RectangleBounds>(1._m, 1._m));

    bucket.push_back(
        makeSpacePoint(SN::BIL, TF::Mdt, false, {100., 5000., 1500.}));
    bucket.push_back(
        makeSpacePoint(SN::BML, TF::Rpc, true, {110., 7000., 2100.}));
    bucket.push_back(
        makeSpacePoint(SN::BOL, TF::Rpc, false, {120., 9500., 2850.}));
    // close to hit 1, so that a line through the two uses the beam spot
    bucket.push_back(
        makeSpacePoint(SN::BML, TF::Mm, false, {112., 7020., 2106.}));
    // same station as hit 1 in another chamber
    bucket.push_back(
        makeSpacePoint(SN::BMS, TF::Rpc, true, {111., 7300., 2190.}));
    bucket.push_back(
        makeSpacePoint(SN::EIL, TF::Mdt, false, {100., 3000., 7000.}));
    bucket.push_back(
        makeSpacePoint(SN::EML, TF::Tgc, true, {110., 4000., 13000.}));
    bucket.push_back(
        makeSpacePoint(SN::EEL, TF::Mdt, false, {100., 3500., 10000.}));
    bucket.push_back(
        makeSpacePoint(SN::BEE, TF::Mdt, false, {100., 4500., 4500.}));

    for (const auto& sp : bucket) {
      payloads.emplace_back(gctx, &sp, &bucket, Transform3::Identity(),
                            surface.get());
    }
    for (const auto& payload : payloads) {
      device.push_back(
          Math::makeHit(ActsExamples::makeCudaHitPayloadRow(gctx, payload)));
    }
  }
};

/// @brief Trigger strip hits of a straight track from the origin in the
///        y-z plane, two layers in each of BI, BM and BO, a little off the
///        line in alternating directions. Then one hit far off the line, and
///        optionally a second hit on the first BM layer, close to the line
struct LineHits {
  ActsExamples::MuonSpacePointBucket bucket{};
  std::shared_ptr<Surface> surface{};
  std::vector<ActsExamples::HitPayload> payloads{};

  static constexpr std::uint32_t nOnLine = 6u;

  LineHits(const GeometryContext& gctx, bool withTwin) {
    using SN = MuonId::StationName;
    // The strips run along x, perpendicular to the bending plane, and measure
    // z
    Transform3 surfaceTrf{Transform3::Identity()};
    surfaceTrf.linear().col(0) = Vector3::UnitZ();
    surfaceTrf.linear().col(1) = Vector3::UnitX();
    surfaceTrf.linear().col(2) = Vector3::UnitY();
    surface = Surface::makeShared<PlaneSurface>(
        surfaceTrf, std::make_shared<RectangleBounds>(1._m, 1._m));

    auto add = [&](SN station, double r, double offset, std::uint8_t layer) {
      auto sp = makeSpacePoint(station, MuonId::TechField::Rpc, false,
                               {0., r, 0.4 * r + offset}, layer);
      sp.setCovariance(0.09, 4., 0.);
      bucket.push_back(sp);
    };
    add(SN::BIL, 4800., 0.1, 1u);
    add(SN::BIL, 5000., -0.1, 2u);
    add(SN::BML, 7000., 0.1, 1u);
    add(SN::BML, 7200., -0.1, 2u);
    add(SN::BOL, 9500., 0.1, 1u);
    add(SN::BOL, 9700., -0.1, 2u);
    add(SN::BML, 7400., 60., 3u);
    if (withTwin) {
      add(SN::BML, 7000., 0.3, 1u);
    }
    for (const auto& sp : bucket) {
      payloads.emplace_back(gctx, &sp, &bucket, Transform3::Identity(),
                            surface.get());
    }
  }
};

/// @brief The pattern finder config of the line tests, and its candidate
///        lists: every hit on the line is a seed and every hit its candidate
ActsExamples::GlobalPatternFinder_t::Config lineConfig() {
  ActsExamples::GlobalPatternFinder_t::Config cfg{};
  cfg.minGroupLayers = 2u;
  return cfg;
}
ActsExamples::CudaCandidateListContainer makeLineLists(const LineHits& hits) {
  std::vector<std::uint32_t> rows(hits.payloads.size());
  std::iota(rows.begin(), rows.end(), 0u);
  ActsExamples::CudaCandidateListContainer lists{};
  // Expanded sector 10 is the centre of sector 5, at phi = 90 degrees
  for (std::uint32_t seed = 0u; seed < LineHits::nOnLine; ++seed) {
    lists.addSeed(seed, 10., 0., rows);
  }
  return lists;
}

/// @brief Runs the seed preparation and the pattern building of the line
ActsExamples::CudaPatternRecordContainer buildLinePatterns(
    const GeometryContext& gctx, const LineHits& hits, std::size_t capacity) {
  auto payloads =
      ActsExamples::makeCudaHitPayloadContainer(gctx, hits.payloads);
  auto lists = makeLineLists(hits);
  ActsExamples::CudaPatternRecordContainer patterns{lists.nSeeds(), capacity,
                                                    16u};
  const auto cudaCfg = ActsExamples::makeCudaPatternFinderConfig(
      lineConfig(), PatternStateAux::BeamspotInfo{});

  ActsExamples::CudaStream stream;
  payloads.moveToDevice(stream.get());
  lists.moveToDevice(stream.get());
  patterns.moveToDevice(stream.get());
  ActsExamples::prepareCudaSeeds(payloads, lists, cudaCfg, stream.get());
  ActsExamples::buildCudaPatterns(payloads, lists, cudaCfg, 0u, patterns,
                                  stream.get());
  patterns.moveToHost(stream.get());
  stream.synchronize();
  return patterns;
}

/// @brief The result of the pattern finding in eta of the line: the hit rows
///        of each accepted pattern, sorted, and what did not fit
struct LineResult {
  std::vector<std::vector<std::uint32_t>> accepted{};
  std::uint8_t overflow{0u};
};

/// @brief Runs the whole pattern finding in eta of the line, with version 1
///        or 2 of the device pattern finding
LineResult findLinePatterns(const GeometryContext& gctx, const LineHits& hits,
                            const ActsExamples::CudaPatternFinderSizes& sizes,
                            unsigned version = 1u) {
  auto payloads =
      ActsExamples::makeCudaHitPayloadContainer(gctx, hits.payloads);
  auto lists = makeLineLists(hits);
  ActsExamples::CudaStream stream;
  payloads.moveToDevice(stream.get());
  lists.moveToDevice(stream.get());
  const auto cudaCfg = ActsExamples::makeCudaPatternFinderConfig(
      lineConfig(), PatternStateAux::BeamspotInfo{});
  const auto event = version == 2u
                         ? ActsExamples::findCudaPatternsInEtaV2(
                               payloads, lists, cudaCfg, sizes, stream.get())
                         : ActsExamples::findCudaPatternsInEta(
                               payloads, lists, cudaCfg, sizes, stream.get());

  LineResult result{.overflow = event.overflow(0u)};
  for (std::uint32_t k = 0u; k < event.nPatterns(0u); ++k) {
    const auto slot = event.slot(0u, k);
    if (event.pattern(slot).status != ActsExamples::CudaPatternStatus::ok) {
      continue;
    }
    auto& rows = result.accepted.emplace_back(event.hitIndices(slot).begin(),
                                              event.hitIndices(slot).end());
    std::ranges::sort(rows);
  }
  return result;
}

/// @brief The rows of the hits on the line
std::vector<std::uint32_t> lineRows() {
  std::vector<std::uint32_t> rows(LineHits::nOnLine);
  std::iota(rows.begin(), rows.end(), 0u);
  return rows;
}

bool hasCudaDevice() {
  int deviceCount = 0;
  return cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0;
}

Math::Vector3 toMath(const Vector3& v) {
  return {v.x(), v.y(), v.z()};
}

/// @brief Bitwise equality of a CPU and a device vector
bool same(const Vector3& cpu, const Math::Vector3& device) {
  return cpu.x() == device.x && cpu.y() == device.y && cpu.z() == device.z;
}

/// @brief The candidate checks of GlobalPatternFinder::findPatternsInEta on
///        one list, written with the CPU types. The sort is stable, as the
///        device sort is
struct CpuSeed {
  std::uint8_t status{ActsExamples::CudaSeedStatus::pending};
  std::vector<std::uint32_t> rows{};
  std::vector<std::uint16_t> globLayer{};
  std::uint16_t seedPosition{0u};
};
CpuSeed prepareSeedOnCpu(
    const std::vector<ActsExamples::HitPayload>& payloads,
    std::vector<std::uint32_t> rows, std::uint32_t seedRow,
    const ActsExamples::GlobalPatternFinder_t::Config& cfg) {
  using ActsExamples::CudaSeedStatus;
  using ActsExamples::PatternTopology;
  CpuSeed out{};
  if (rows.size() < cfg.minStripEtaLayers + cfg.minPrecisionLayers) {
    out.status = CudaSeedStatus::tooFewHits;
    return out;
  }
  std::array<PatternTopology::GroupIdx, PatternTopology::nGroups> groupCounts{};
  PatternTopology::GroupIdx nValidGroups{0u};
  for (const std::uint32_t row : rows) {
    if (++groupCounts[PatternTopology::groupIndex(payloads[row])] ==
        cfg.minGroupLayers) {
      ++nValidGroups;
    }
    if (nValidGroups >= cfg.minGroups) {
      break;
    }
  }
  if (nValidGroups < cfg.minGroups) {
    out.status = CudaSeedStatus::tooFewGroups;
    return out;
  }
  std::ranges::stable_sort(rows, [&](std::uint32_t r1, std::uint32_t r2) {
    return PatternTopology::layerSorter(payloads[r1], payloads[r2]);
  });
  out.globLayer.push_back(0u);
  for (std::size_t i = 1; i < rows.size(); ++i) {
    out.globLayer.push_back(
        out.globLayer.back() +
        !PatternTopology::sameLayer(payloads[rows[i - 1]], payloads[rows[i]]));
  }
  out.rows = rows;
  if (out.globLayer.back() + 1u <
      cfg.minStripEtaLayers + cfg.minPrecisionLayers) {
    out.status = CudaSeedStatus::tooFewLayers;
    return out;
  }
  out.status = CudaSeedStatus::accepted;
  out.seedPosition = static_cast<std::uint16_t>(
      std::ranges::find(rows, seedRow) - rows.begin());
  return out;
}
}  // namespace

BOOST_AUTO_TEST_SUITE(CudaPatternFinderSuite)

BOOST_AUTO_TEST_CASE(CudaPatternLineMatchesCpu) {
  using BendPlaneCov = PatternStateAux::BendPlaneCov;
  const auto gctx = GeometryContext::dangerouslyDefaultConstruct();
  const Hits hits{gctx};
  const auto logger = getDefaultLogger("CudaPatternMath", Logging::WARNING);
  const PatternStateAux::Config cfg{};
  const PatternStateAux::BeamspotInfo beamspot{Vector3{0., 0., 10.}, 1., 100.};
  const Math::Beamspot deviceBeamspot{toMath(beamspot.position),
                                      beamspot.radius, beamspot.length};

  // The variance of every hit along a few directions, projected and not
  for (std::size_t h = 0; h < hits.payloads.size(); ++h) {
    for (const Vector3& c :
         {Vector3{1., 0., 0.}, Vector3{1., 2., 3.}.normalized()}) {
      for (const bool projected : {false, true}) {
        BOOST_TEST_CONTEXT("hit " << h << ", projected " << projected) {
          BOOST_CHECK_EQUAL(
              PatternStateAux::intrinsicVariance(gctx, hits.payloads[h], c,
                                                 projected),
              Math::intrinsicVariance(hits.device[h], toMath(c), projected));
        }
      }
    }
  }

  // Anchor and last hit far apart, then close enough to need the beam spot
  for (const auto [anchor, last] : {std::pair{0u, 2u}, std::pair{1u, 3u}}) {
    PatternStateAux pat{gctx, OrderedHit{&hits.payloads[anchor], 0u},
                        4,    beamspot,
                        &cfg, logger.get()};
    // A bending plane off the origin, with a correlated covariance
    pat.patPhi = std::atan2(hits.payloads[1].position.y(),
                            hits.payloads[1].position.x());
    pat.bendPlaneNorm =
        Vector3{-std::sin(pat.patPhi), std::cos(pat.patPhi), 0.};
    pat.patPhiOffset = 25.;
    pat.bendPlaneCov[Acts::toUnderlying(BendPlaneCov::ePhiPhi)] = 1.e-4;
    pat.bendPlaneCov[Acts::toUnderlying(BendPlaneCov::eSS)] = 4.;
    pat.bendPlaneCov[Acts::toUnderlying(BendPlaneCov::ePhiS)] = -1.e-2;
    pat.lineAnchorHit = OrderedHit{&hits.payloads[anchor], 0u};
    pat.lastInsertedHit = OrderedHit{&hits.payloads[last], 5u};
    pat.needLineUpdate = true;
    pat.updateLineParameters(gctx, beamspot);

    const Math::BendPlane plane{
        .phi = pat.patPhi,
        .offset = pat.patPhiOffset,
        .normal = toMath(pat.bendPlaneNorm),
        .covPhiPhi =
            pat.bendPlaneCov[Acts::toUnderlying(BendPlaneCov::ePhiPhi)],
        .covSS = pat.bendPlaneCov[Acts::toUnderlying(BendPlaneCov::eSS)],
        .covPhiS = pat.bendPlaneCov[Acts::toUnderlying(BendPlaneCov::ePhiS)]};
    const Math::Line line{Math::lineThroughHits(
        Math::projectToPhiPlane(hits.device[anchor], plane.normal,
                                plane.offset),
        Math::projectToPhiPlane(hits.device[last], plane.normal, plane.offset),
        cfg.minHitDistance4Line, deviceBeamspot)};

    BOOST_TEST_CONTEXT("anchor " << anchor << ", last " << last) {
      BOOST_CHECK(same(pat.projToPhiPlane(gctx, hits.payloads[last]),
                       Math::projectToPhiPlane(hits.device[last], plane.normal,
                                               plane.offset)));
      BOOST_CHECK_EQUAL(pat.useBeamspot, line.useBeamspot);
      BOOST_CHECK(same(pat.linePos, line.position));
      BOOST_CHECK(same(pat.lineDir, line.direction));
      BOOST_CHECK_EQUAL(pat.leverArm, line.leverArm);
      BOOST_CHECK_EQUAL(pat.angleToBendPlane(hits.payloads[6].position),
                        Math::angleToBendPlane(plane, hits.device[6].position));

      // The residual of every other hit against the line, with and without a
      // phi measurement in the pattern
      for (unsigned nPhiLayers : {0u, 1u}) {
        pat.nPhiLayers = nPhiLayers;
        for (std::size_t test = 0; test < hits.payloads.size(); ++test) {
          const auto cpu = pat.computeLineResidual(
              gctx, OrderedHit{&hits.payloads[test], 6u}, beamspot);
          const auto device = Math::lineResidual(
              line, plane, nPhiLayers, hits.device[anchor], hits.device[last],
              hits.device[test], deviceBeamspot);
          BOOST_CHECK_EQUAL(cpu.residual, device.residual);
          BOOST_CHECK_EQUAL(cpu.sigma, device.sigma);
          BOOST_CHECK_EQUAL(static_cast<int>(cpu.nDof),
                            static_cast<int>(device.nDof));
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(CudaBendPlaneFitMatchesCpu) {
  using BendPlaneCov = PatternStateAux::BendPlaneCov;
  const auto gctx = GeometryContext::dangerouslyDefaultConstruct();
  const Hits hits{gctx};
  const auto logger = getDefaultLogger("CudaPatternMath", Logging::WARNING);
  const PatternStateAux::Config cfg{};
  const PatternStateAux::BeamspotInfo beamspot{Vector3{1., -2., 10.}, 2., 100.};
  const Math::Beamspot deviceBeamspot{toMath(beamspot.position),
                                      beamspot.radius, beamspot.length};
  constexpr std::int8_t sector{10};

  // No phi hit, one, then the three phi hits of the fixture (rows 1, 4, 6)
  for (const std::vector<std::uint32_t>& phiRows :
       {std::vector<std::uint32_t>{}, std::vector<std::uint32_t>{4u},
        std::vector<std::uint32_t>{1u, 4u, 6u}}) {
    PatternStateAux pat{gctx,   OrderedHit{&hits.payloads[0], 0u},
                        sector, beamspot,
                        &cfg,   logger.get()};
    for (const std::uint32_t row : phiRows) {
      pat.hitsPerGroup[ActsExamples::PatternTopology::groupIndex(
                           hits.payloads[row])]
          .push_back(OrderedHit{&hits.payloads[row], 0u});
    }
    pat.nPhiLayers = phiRows.size();
    const double previousPhi{0.3};
    pat.patPhi = previousPhi;
    pat.updatePatternPhi(gctx, beamspot);

    // The device visits the same hits, group by group
    auto forEachPhiHit = [&](auto&& fn) {
      for (std::uint32_t g = 0u; g < Math::Station::nGroups; ++g) {
        for (const std::uint32_t row : phiRows) {
          if (Math::groupIndex(hits.device[row]) == g) {
            fn(hits.device[row]);
          }
        }
      }
    };
    const Math::BendPlane plane{
        Math::fitBendPlane(static_cast<std::uint32_t>(phiRows.size()),
                           previousPhi, sector, deviceBeamspot, forEachPhiHit)};

    BOOST_TEST_CONTEXT(phiRows.size() << " phi hits") {
      BOOST_CHECK_EQUAL(pat.patPhi, plane.phi);
      BOOST_CHECK_EQUAL(pat.patPhiOffset, plane.offset);
      BOOST_CHECK(same(pat.bendPlaneNorm, plane.normal));
      BOOST_CHECK_EQUAL(
          pat.bendPlaneCov[Acts::toUnderlying(BendPlaneCov::ePhiPhi)],
          plane.covPhiPhi);
      BOOST_CHECK_EQUAL(pat.bendPlaneCov[Acts::toUnderlying(BendPlaneCov::eSS)],
                        plane.covSS);
      BOOST_CHECK_EQUAL(
          pat.bendPlaneCov[Acts::toUnderlying(BendPlaneCov::ePhiS)],
          plane.covPhiS);
    }
  }
}

BOOST_AUTO_TEST_CASE(CudaPatternLayerOrderMatchesCpu) {
  using ActsExamples::PatternTopology;
  const auto gctx = GeometryContext::dangerouslyDefaultConstruct();
  const Hits hits{gctx};

  for (std::size_t i = 0; i < hits.payloads.size(); ++i) {
    for (std::size_t j = 0; j < hits.payloads.size(); ++j) {
      BOOST_TEST_CONTEXT("hits " << i << " and " << j) {
        BOOST_CHECK_EQUAL(
            PatternTopology::layerSorter(hits.payloads[i], hits.payloads[j]),
            Math::layerSorter(hits.device[i], hits.device[j]));
        BOOST_CHECK_EQUAL(
            PatternTopology::sameLayer(hits.payloads[i], hits.payloads[j]),
            Math::sameLayer(hits.device[i], hits.device[j]));
      }
    }
    BOOST_CHECK_EQUAL(
        static_cast<unsigned>(PatternTopology::groupIndex(hits.payloads[i])),
        static_cast<unsigned>(Math::groupIndex(hits.device[i])));
  }
}

BOOST_AUTO_TEST_CASE(CudaSeedPreparationMatchesCpu) {
  using ActsExamples::CudaSeedStatus;
  if (!hasCudaDevice()) {
    BOOST_TEST_MESSAGE("No CUDA device found, skipping CUDA runtime test");
    return;
  }
  const auto gctx = GeometryContext::dangerouslyDefaultConstruct();
  const Hits hits{gctx};
  ActsExamples::GlobalPatternFinder_t::Config cfg{};
  cfg.minGroups = 1u;
  cfg.minGroupLayers = 2u;

  // One seed per outcome. Rows 1, 3 and 4 are the three BM hits
  struct Seed {
    std::uint32_t seedRow;
    std::vector<std::uint32_t> rows;
    std::uint8_t expected;
  };
  const std::vector<Seed> seeds{
      {2u, {8u, 7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u}, CudaSeedStatus::accepted},
      {1u, {1u, 3u}, CudaSeedStatus::tooFewHits},
      {0u, {0u, 2u, 5u, 6u}, CudaSeedStatus::tooFewGroups},
      {1u, {1u, 1u, 1u}, CudaSeedStatus::tooFewLayers},
      {3u, {4u, 3u, 1u, 0u, 2u}, CudaSeedStatus::accepted}};

  auto payloads =
      ActsExamples::makeCudaHitPayloadContainer(gctx, hits.payloads);
  ActsExamples::CudaCandidateListContainer lists{};
  for (const Seed& seed : seeds) {
    lists.addSeed(seed.seedRow, 0., 0., seed.rows);
  }
  ActsExamples::CudaStream stream;
  payloads.moveToDevice(stream.get());
  lists.moveToDevice(stream.get());
  ActsExamples::prepareCudaSeeds(payloads, lists,
                                 ActsExamples::makeCudaPatternFinderConfig(
                                     cfg, PatternStateAux::BeamspotInfo{}),
                                 stream.get());
  lists.moveToHost(stream.get());
  stream.synchronize();

  for (std::size_t s = 0; s < seeds.size(); ++s) {
    const CpuSeed cpu{
        prepareSeedOnCpu(hits.payloads, seeds[s].rows, seeds[s].seedRow, cfg)};
    BOOST_TEST_CONTEXT("seed " << s) {
      BOOST_CHECK_EQUAL(cpu.status, seeds[s].expected);
      BOOST_CHECK_EQUAL(lists.seedStatus(s), cpu.status);
      if (cpu.status == CudaSeedStatus::accepted) {
        BOOST_CHECK_EQUAL(lists.seedPosition(s), cpu.seedPosition);
      }
      if (!cpu.rows.empty()) {
        const auto rows = lists.candidateHitIndices(s);
        const auto globLayer = lists.candidateGlobLayers(s);
        BOOST_CHECK_EQUAL_COLLECTIONS(rows.begin(), rows.end(),
                                      cpu.rows.begin(), cpu.rows.end());
        BOOST_CHECK_EQUAL_COLLECTIONS(globLayer.begin(), globLayer.end(),
                                      cpu.globLayer.begin(),
                                      cpu.globLayer.end());
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(CudaExpandedSectorMatchesCpu) {
  namespace Sector = Math::ExpandedSector;
  using ActsExamples::ExpandedSector;
  for (int s = 0; s < Sector::nExpanded; ++s) {
    const ExpandedSector cpu{static_cast<ExpandedSector::Index_t>(s)};
    BOOST_TEST_CONTEXT("sector " << s) {
      BOOST_CHECK_EQUAL(cpu.phi(), Sector::phi(s));
      BOOST_CHECK_EQUAL(cpu.sectorSize(), Sector::size(s));
      for (int o = 0; o < Sector::nExpanded; ++o) {
        BOOST_CHECK_EQUAL(cpu.isNeighbour(ExpandedSector{
                              static_cast<ExpandedSector::Index_t>(o)}),
                          Sector::isNeighbour(s, o));
      }
      for (int i = 0; i < 720; ++i) {
        const double phi{-Math::pi + i * Math::pi / 360.};
        BOOST_CHECK_EQUAL(cpu.insideSector(phi), Sector::inside(s, phi));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(CudaPatternBuildingFindsLine) {
  if (!hasCudaDevice()) {
    BOOST_TEST_MESSAGE("No CUDA device found, skipping CUDA runtime test");
    return;
  }
  const auto gctx = GeometryContext::dangerouslyDefaultConstruct();
  const LineHits hits{gctx, false};
  const auto patterns = buildLinePatterns(gctx, hits, 4u);

  // The seeds on the BI and BM layers find the line, without the outlier.
  // From a BO seed the line misses the minimum layer requirement, as in Core
  for (std::uint32_t seed = 0u; seed < LineHits::nOnLine; ++seed) {
    BOOST_TEST_CONTEXT("seed " << seed) {
      BOOST_CHECK(!patterns.overflow(seed));
      if (seed >= LineHits::nOnLine - 2u) {
        BOOST_CHECK_EQUAL(patterns.nPatterns(seed), 0u);
        continue;
      }
      BOOST_REQUIRE_EQUAL(patterns.nPatterns(seed), 1u);
      const auto slot = patterns.slot(seed, 0u);
      std::vector<std::uint32_t> found{patterns.hitIndices(slot).begin(),
                                       patterns.hitIndices(slot).end()};
      std::ranges::sort(found);
      const std::vector<std::uint32_t> expected{lineRows()};
      BOOST_CHECK_EQUAL_COLLECTIONS(found.begin(), found.end(),
                                    expected.begin(), expected.end());
      BOOST_CHECK_EQUAL(patterns.pattern(slot).nTriggerLayers,
                        LineHits::nOnLine);
    }
  }
}

BOOST_AUTO_TEST_CASE(CudaPatternBuildingOverflow) {
  if (!hasCudaDevice()) {
    BOOST_TEST_MESSAGE("No CUDA device found, skipping CUDA runtime test");
    return;
  }
  const auto gctx = GeometryContext::dangerouslyDefaultConstruct();
  // The second hit on the first BM layer branches the pattern of the first
  // seed in two
  const LineHits hits{gctx, true};

  const auto small = buildLinePatterns(gctx, hits, 1u);
  BOOST_CHECK(small.overflow(0u));
  BOOST_CHECK_EQUAL(small.nPatterns(0u), 0u);

  // With room for both branches, the overlap removal keeps one of them
  const auto large = buildLinePatterns(gctx, hits, 4u);
  BOOST_CHECK(!large.overflow(0u));
  BOOST_CHECK_EQUAL(large.nPatterns(0u), 1u);
}

BOOST_AUTO_TEST_CASE(CudaPatternFindingAcceptsLineOnce) {
  if (!hasCudaDevice()) {
    BOOST_TEST_MESSAGE("No CUDA device found, skipping CUDA runtime test");
    return;
  }
  const auto gctx = GeometryContext::dangerouslyDefaultConstruct();
  const LineHits hits{gctx, false};
  const std::vector<std::uint32_t> expected{lineRows()};

  // All six seeds find the line; it is accepted once, whatever the batches,
  // in both versions
  for (const unsigned version : {1u, 2u}) {
    for (const std::size_t batchSize : {1u, 4u, 64u}) {
      BOOST_TEST_CONTEXT("version " << version << ", batch size "
                                    << batchSize) {
        const auto accepted =
            findLinePatterns(gctx, hits, {.batchSize = batchSize}, version)
                .accepted;
        BOOST_REQUIRE_EQUAL(accepted.size(), 1u);
        BOOST_CHECK_EQUAL_COLLECTIONS(accepted[0].begin(), accepted[0].end(),
                                      expected.begin(), expected.end());
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(CudaPatternFindingReportsOverflow) {
  if (!hasCudaDevice()) {
    BOOST_TEST_MESSAGE("No CUDA device found, skipping CUDA runtime test");
    return;
  }
  const auto gctx = GeometryContext::dangerouslyDefaultConstruct();
  const LineHits hits{gctx, true};

  for (const unsigned version : {1u, 2u}) {
    BOOST_TEST_CONTEXT("version " << version) {
      // A workspace with room for both branches of every seed: the line is
      // found and nothing overflows
      const auto ample = findLinePatterns(gctx, hits, {}, version);
      BOOST_CHECK_EQUAL(ample.overflow, 0u);
      BOOST_CHECK_EQUAL(ample.accepted.size(), 1u);

      // A workspace with room for one pattern per region and seed: the seeds
      // whose pattern branches have no patterns and are reported, and the
      // seed on the layer of the second hit, which does not branch, still
      // finds the line
      const auto small =
          findLinePatterns(gctx, hits, {.workspaceBytes = 1u}, version);
      BOOST_CHECK_EQUAL(small.overflow,
                        ActsExamples::CudaPatternOverflow::patterns);
      BOOST_CHECK_EQUAL(small.accepted.size(), 1u);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
