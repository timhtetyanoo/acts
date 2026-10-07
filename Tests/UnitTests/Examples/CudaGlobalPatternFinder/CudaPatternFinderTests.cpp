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
#include <vector>

using namespace Acts::UnitLiterals;

namespace ActsTests {

using namespace Acts;
namespace Math = ActsExamples::CudaPatternMath;

using MuonId = ActsExamples::MuonSpacePoint::MuonId;
using PatternState = ActsExamples::GlobalPatternFinder_t::PatternState;
using OrderedHit = PatternState::OrderedHit;

namespace {
ActsExamples::MuonSpacePoint makeSpacePoint(MuonId::StationName station,
                                            MuonId::TechField tech,
                                            bool measPhi, const Vector3& pos) {
  MuonId id{};
  id.setChamber(station, MuonId::DetSide::A, 3u, tech);
  id.setLayAndCh(2u, 5u);
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
  const auto gctx = GeometryContext::dangerouslyDefaultConstruct();
  const Hits hits{gctx};
  const auto logger = getDefaultLogger("CudaPatternMath", Logging::WARNING);
  const PatternState::Config cfg{};
  const PatternState::BeamspotInfo beamspot{Vector3{0., 0., 10.}, 1., 100.};
  const Math::Beamspot deviceBeamspot{toMath(beamspot.position),
                                      beamspot.radius, beamspot.length};

  // Anchor and last hit far apart, then close enough to need the beam spot
  for (const auto [anchor, last] : {std::pair{0u, 2u}, std::pair{1u, 3u}}) {
    PatternState pat{gctx, OrderedHit{&hits.payloads[anchor], 0u}, 4, &cfg,
                     logger.get()};
    const double phi{std::atan2(hits.payloads[1].position.y(),
                                hits.payloads[1].position.x())};
    pat.bendPlaneNorm = Vector3{-std::sin(phi), std::cos(phi), 0.};
    pat.lineAnchorHit = OrderedHit{&hits.payloads[anchor], 0u};
    pat.lastInsertedHit = OrderedHit{&hits.payloads[last], 5u};
    pat.needLineUpdate = true;
    pat.updateLineParameters(gctx, beamspot);

    const Math::Vector3 norm{toMath(pat.bendPlaneNorm)};
    const Math::Line line{Math::lineThroughHits(
        Math::projectToPhiPlane(hits.device[anchor], norm),
        Math::projectToPhiPlane(hits.device[last], norm),
        cfg.minHitDistance4Line, deviceBeamspot)};

    BOOST_TEST_CONTEXT("anchor " << anchor << ", last " << last) {
      BOOST_CHECK(same(pat.projToPhiPlane(gctx, hits.payloads[last]),
                       Math::projectToPhiPlane(hits.device[last], norm)));
      BOOST_CHECK_EQUAL(pat.useBeamspot, line.useBeamspot);
      BOOST_CHECK(same(pat.linePos, line.position));
      BOOST_CHECK(same(pat.lineDir, line.direction));
      BOOST_CHECK_EQUAL(pat.leverArm, line.leverArm);

      // The residual of every other hit against the line, with and without a
      // phi measurement in the pattern
      for (unsigned nPhiLayers : {0u, 1u}) {
        pat.nPhiLayers = nPhiLayers;
        pat.patPhiCov = 1.e-4;
        for (std::size_t test = 0; test < hits.payloads.size(); ++test) {
          const auto cpu = pat.computeLineResidual(
              gctx, OrderedHit{&hits.payloads[test], 6u}, beamspot);
          const auto device = Math::lineResidual(
              line, norm, pat.patPhiCov, nPhiLayers, hits.device[anchor],
              hits.device[last], hits.device[test], deviceBeamspot);
          BOOST_CHECK_EQUAL(cpu.residual, device.residual);
          BOOST_CHECK_EQUAL(cpu.sigma, device.sigma);
        }
      }
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
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
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
                                 ActsExamples::makeCudaPatternFinderConfig(cfg),
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

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
