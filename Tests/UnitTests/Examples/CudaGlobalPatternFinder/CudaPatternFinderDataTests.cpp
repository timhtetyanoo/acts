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
#include "Acts/Utilities/MathHelpers.hpp"
#include "ActsExamples/EventData/CudaPatternFinderData.hpp"
#include "ActsExamples/EventData/MuonSpacePoint.hpp"
#include "ActsExamples/TrackFinding/GlobalPatternFinderAlgorithm.hpp"
#include "ActsExamples/Utilities/CudaStream.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>

#include <cuda_runtime.h>

using namespace Acts::UnitLiterals;

namespace ActsTests {

using namespace Acts;

using MuonId = ActsExamples::MuonSpacePoint::MuonId;
using ActsExamples::CudaHitFlags;

namespace {
/// @brief Build an identifier of a chamber with the given station fields
MuonId makeId(MuonId::StationName station, MuonId::DetSide side,
              std::uint16_t sector) {
  MuonId id{};
  id.setChamber(station, side, sector, MuonId::TechField::Mdt);
  id.setLayAndCh(1, 1);
  id.setCoordFlags(false, true, true);

  return id;
}

/// @brief Build a space point with the given technology and measured coordinates
ActsExamples::MuonSpacePoint makeSpacePoint(
    MuonId::TechField tech, bool measPhi, const Vector3& pos,
    MuonId::StationName station = MuonId::StationName::BML) {
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

}  // namespace

BOOST_AUTO_TEST_SUITE(EventDataSuite)

BOOST_AUTO_TEST_CASE(CudaHitPayloadHostConstruction) {
  ActsExamples::CudaHitPayloadContainer payloads{2};

  BOOST_CHECK_EQUAL(payloads.size(), 2u);
  BOOST_CHECK(!payloads.empty());
  BOOST_CHECK(!payloads.isOnDevice());

  payloads.setHit(0, Vector3{1.0, 2.0, 3.0}, 4.0, 5u, 6u, 7u);

  BOOST_CHECK_EQUAL(payloads.position(0).x(), 1.0);
  BOOST_CHECK_EQUAL(payloads.position(0).y(), 2.0);
  BOOST_CHECK_EQUAL(payloads.position(0).z(), 3.0);
  BOOST_CHECK_EQUAL(payloads.localPositionY(0), 4.0);
  BOOST_CHECK_EQUAL(payloads.stationKey(0), 5u);
  BOOST_CHECK_EQUAL(payloads.station(0), 6u);
  BOOST_CHECK_EQUAL(payloads.locLayer(0), 7u);

  BOOST_CHECK_EQUAL(payloads.position(1).x(), 0.0);
  BOOST_CHECK_EQUAL(payloads.stationKey(1), 0u);

  BOOST_CHECK_THROW(payloads.setHit(2, Vector3::Zero(), 0.0, 0u, 0u, 0u),
                    std::out_of_range);
  BOOST_CHECK_THROW(payloads.station(2), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(CudaHitPayloadStationKey) {
  using StationName = MuonId::StationName;
  using DetSide = MuonId::DetSide;

  const MuonId reference{makeId(StationName::BIS, DetSide::A, 3)};

  // Identifiers of the same chamber share the key
  BOOST_CHECK_EQUAL(
      ActsExamples::stationKey(reference),
      ActsExamples::stationKey(makeId(StationName::BIS, DetSide::A, 3)));

  // Each of the three station fields separates the keys
  BOOST_CHECK_NE(
      ActsExamples::stationKey(reference),
      ActsExamples::stationKey(makeId(StationName::BIL, DetSide::A, 3)));
  BOOST_CHECK_NE(
      ActsExamples::stationKey(reference),
      ActsExamples::stationKey(makeId(StationName::BIS, DetSide::C, 3)));
  BOOST_CHECK_NE(
      ActsExamples::stationKey(reference),
      ActsExamples::stationKey(makeId(StationName::BIS, DetSide::A, 4)));

  // The key agrees with the identifier on sharing a station
  const MuonId other{makeId(StationName::BIS, DetSide::A, 4)};
  BOOST_CHECK_EQUAL(
      reference.sameStation(other),
      ActsExamples::stationKey(reference) == ActsExamples::stationKey(other));
}

BOOST_AUTO_TEST_CASE(CudaHitPayloadDeviceRoundTrip) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    BOOST_TEST_MESSAGE("No CUDA device found, skipping CUDA runtime test");
    return;
  }

  ActsExamples::CudaHitPayloadContainer payloads{1};
  ActsExamples::CudaStream stream;

  payloads.setHit(0, Vector3{1.0, 2.0, 3.0}, 4.0,
                  ActsExamples::stationKey(
                      makeId(MuonId::StationName::BIS, MuonId::DetSide::A, 3)),
                  6u, 7u);

  payloads.moveToDevice(stream.get());
  stream.synchronize();

  BOOST_CHECK(payloads.isOnDevice());
  BOOST_CHECK(payloads.deviceArrays().positionX != nullptr);
  BOOST_CHECK(payloads.deviceArrays().localPositionY != nullptr);
  BOOST_CHECK(payloads.deviceArrays().stationKey != nullptr);
  BOOST_CHECK(payloads.deviceArrays().station != nullptr);
  BOOST_CHECK(payloads.deviceArrays().locLayer != nullptr);

  const std::uint32_t uploadedKey{payloads.stationKey(0)};

  // Overwrite the host columns, so the copy back is the only source of the
  // values checked below
  payloads.setHit(0, Vector3::Zero(), 0.0, 0u, 0u, 0u);

  payloads.moveToHost(stream.get());
  stream.synchronize();

  BOOST_CHECK_EQUAL(payloads.position(0).x(), 1.0);
  BOOST_CHECK_EQUAL(payloads.position(0).y(), 2.0);
  BOOST_CHECK_EQUAL(payloads.position(0).z(), 3.0);
  BOOST_CHECK_EQUAL(payloads.localPositionY(0), 4.0);
  BOOST_CHECK_EQUAL(payloads.stationKey(0), uploadedKey);
  BOOST_CHECK_EQUAL(payloads.station(0), 6u);
  BOOST_CHECK_EQUAL(payloads.locLayer(0), 7u);

  payloads.clearDevice();

  BOOST_CHECK(!payloads.isOnDevice());
  BOOST_CHECK(payloads.deviceArrays().positionX == nullptr);
}

BOOST_AUTO_TEST_CASE(CudaHitPayloadRowRoundTrip) {
  using ActsExamples::CudaHitFlags;

  ActsExamples::CudaHitPayloadRow row{};
  row.position = Vector3{1.0, 2.0, 3.0};
  row.sensorDirection = Vector3{0.0, 1.0, 0.0};
  row.etaMeasurementDirection = Vector3{1.0, 0.0, 0.0};
  row.phiMeasurementDirection = Vector3{0.0, 0.0, 1.0};
  row.localPositionY = 4.0;
  row.etaCovariance = 0.09;
  row.phiCovariance = 4.0;
  row.driftCovariance = 0.5;
  row.geometryId = 11u;
  row.stationKey = 5u;
  row.station = 6u;
  row.locLayer = 7u;
  row.flags = CudaHitFlags::straw | CudaHitFlags::precision;

  ActsExamples::CudaHitPayloadContainer payloads{2};
  payloads.setHit(1, row);

  // Only the row that was set holds values
  BOOST_CHECK(payloads.hit(1).sensorDirection == row.sensorDirection);
  BOOST_CHECK_EQUAL(payloads.hit(1).driftCovariance, 0.5);
  BOOST_CHECK_EQUAL(payloads.hit(1).geometryId, 11u);
  BOOST_CHECK_EQUAL(payloads.hit(1).flags, row.flags);
  BOOST_CHECK_EQUAL(payloads.hit(0).geometryId, 0u);
  BOOST_CHECK_EQUAL(payloads.hit(0).flags, 0u);

  // The setter that takes a few quantities clears the others
  payloads.setHit(1, Vector3{1.0, 2.0, 3.0}, 4.0, 5u, 6u, 7u);
  BOOST_CHECK_EQUAL(payloads.hit(1).driftCovariance, 0.0);
  BOOST_CHECK_EQUAL(payloads.hit(1).flags, 0u);
  payloads.setHit(1, row);

  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    BOOST_TEST_MESSAGE("No CUDA device found, skipping CUDA runtime test");
    return;
  }

  ActsExamples::CudaStream stream;
  payloads.moveToDevice(stream.get());
  stream.synchronize();

  // Overwrite the host columns, so the copy back is the only source of the
  // values checked below
  payloads.setHit(1, ActsExamples::CudaHitPayloadRow{});

  payloads.moveToHost(stream.get());
  stream.synchronize();

  const ActsExamples::CudaHitPayloadRow back{payloads.hit(1)};
  BOOST_CHECK(back.position == row.position);
  BOOST_CHECK(back.sensorDirection == row.sensorDirection);
  BOOST_CHECK(back.etaMeasurementDirection == row.etaMeasurementDirection);
  BOOST_CHECK(back.phiMeasurementDirection == row.phiMeasurementDirection);
  BOOST_CHECK_EQUAL(back.etaCovariance, row.etaCovariance);
  BOOST_CHECK_EQUAL(back.phiCovariance, row.phiCovariance);
  BOOST_CHECK_EQUAL(back.driftCovariance, row.driftCovariance);
  BOOST_CHECK_EQUAL(back.geometryId, row.geometryId);
  BOOST_CHECK_EQUAL(back.flags, row.flags);
}

BOOST_AUTO_TEST_CASE(CudaHitPayloadRowFromHitPayload) {
  const auto gctx = GeometryContext::dangerouslyDefaultConstruct();

  // A surface that is neither at the origin nor aligned with the global axes
  Transform3 surfaceTrf{Transform3::Identity()};
  surfaceTrf.translate(Vector3{10., 20., 30.});
  surfaceTrf.rotate(AngleAxis3{0.3, Vector3::UnitZ()});
  surfaceTrf.rotate(AngleAxis3{0.2, Vector3::UnitX()});
  const auto surface = Surface::makeShared<PlaneSurface>(
      surfaceTrf, std::make_shared<RectangleBounds>(1._m, 1._m));

  ActsExamples::MuonSpacePointBucket bucket{};
  bucket.push_back(
      makeSpacePoint(MuonId::TechField::Mdt, false, Vector3{5., 11., 130.}));
  bucket.push_back(
      makeSpacePoint(MuonId::TechField::Rpc, true, Vector3{6., 12., 140.}));

  const Transform3 toGlobal{Transform3::Identity()};
  const ActsExamples::HitPayload straw{gctx, &bucket[0], &bucket, toGlobal,
                                       surface.get()};
  const ActsExamples::HitPayload strip{gctx, &bucket[1], &bucket, toGlobal,
                                       surface.get()};

  const auto strawRow = ActsExamples::makeCudaHitPayloadRow(gctx, straw);
  const auto stripRow = ActsExamples::makeCudaHitPayloadRow(gctx, strip);

  BOOST_CHECK(strawRow.position == straw.globalPosition(gctx));
  BOOST_CHECK(strawRow.sensorDirection == straw.globalSensorDirection(gctx));
  BOOST_CHECK_EQUAL(strawRow.geometryId, 7u);
  BOOST_CHECK_EQUAL(strawRow.flags, CudaHitFlags::straw |
                                        CudaHitFlags::measuresEta |
                                        CudaHitFlags::precision);
  BOOST_CHECK_EQUAL(stripRow.flags,
                    CudaHitFlags::measuresEta | CudaHitFlags::measuresPhi);
  // What PatternStateAux::intrinsicVariance reads: the covariance in the
  // ACTS order (non-bending at 0, bending at 1), the drift circle of the straw
  // and the measured directions of the strip
  const auto& strawCov = bucket[0].covariance();
  const auto& stripCov = bucket[1].covariance();
  BOOST_CHECK_EQUAL(strawRow.driftCovariance,
                    square(bucket[0].driftRadius()) + strawCov[1]);
  BOOST_CHECK_EQUAL(strawRow.phiCovariance, strawCov[0]);
  BOOST_CHECK_EQUAL(stripRow.etaCovariance, stripCov[1]);
  BOOST_CHECK_EQUAL(stripRow.phiCovariance, stripCov[0]);
  const auto [primary, secondary] = strip.stripMeasurementDirections(gctx);
  BOOST_REQUIRE(secondary.has_value());
  BOOST_CHECK(stripRow.etaMeasurementDirection == primary);
  BOOST_CHECK(stripRow.phiMeasurementDirection == *secondary);
}

BOOST_AUTO_TEST_CASE(CudaCandidateListHostConstruction) {
  ActsExamples::CudaCandidateListContainer lists;

  BOOST_CHECK(lists.empty());
  BOOST_CHECK_EQUAL(lists.nSeeds(), 0u);
  BOOST_CHECK_EQUAL(lists.nCandidates(), 0u);
  BOOST_CHECK(!lists.isOnDevice());

  lists.addSeed(7u, 3.0, 0.40, {10u, 11u, 12u});
  lists.addSeed(11u, 3.0, 0.41, {11u, 20u});
  lists.addSeed(20u, 4.0, 0.90, {});

  BOOST_CHECK(!lists.empty());
  BOOST_CHECK_EQUAL(lists.nSeeds(), 3u);
  BOOST_CHECK_EQUAL(lists.nCandidates(), 5u);

  BOOST_CHECK_EQUAL(lists.seedHitIndex(0), 7u);
  BOOST_CHECK_EQUAL(lists.seedSector(0), 3.0);
  BOOST_CHECK_EQUAL(lists.seedTheta(0), 0.40);
  BOOST_CHECK_EQUAL(lists.nCandidates(0), 3u);
  BOOST_CHECK_EQUAL(lists.candidateBegin(0), 0u);
  BOOST_CHECK_EQUAL(lists.candidateEnd(0), 3u);
  BOOST_CHECK_EQUAL(lists.candidateHitIndex(0, 1), 11u);

  // Two seeds can own the same payload row, as overlapping tree searches do
  BOOST_CHECK_EQUAL(lists.candidateHitIndex(0, 1),
                    lists.candidateHitIndex(1, 0));
  BOOST_CHECK_EQUAL(lists.nCandidates(1), 2u);
  BOOST_CHECK_EQUAL(lists.candidateBegin(1), 3u);
  BOOST_CHECK_EQUAL(lists.candidateEnd(1), 5u);
  BOOST_CHECK_EQUAL(lists.seedHitIndex(1), 11u);

  BOOST_CHECK_EQUAL(lists.nCandidates(2), 0u);
  BOOST_CHECK_EQUAL(lists.candidateBegin(2), 5u);
  BOOST_CHECK_EQUAL(lists.candidateEnd(2), 5u);
  BOOST_CHECK(lists.candidateHitIndices(2).empty());

  const std::span<const std::uint32_t> first{lists.candidateHitIndices(0)};
  BOOST_CHECK_EQUAL(first.size(), 3u);
  BOOST_CHECK_EQUAL(first[0], 10u);
  BOOST_CHECK_EQUAL(first[2], 12u);

  BOOST_CHECK_THROW(lists.seedHitIndex(3), std::out_of_range);
  BOOST_CHECK_THROW(lists.candidateHitIndex(0, 3), std::out_of_range);
  BOOST_CHECK_THROW(lists.setCandidate(5, 0u), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(CudaCandidateListDeviceRoundTrip) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    BOOST_TEST_MESSAGE("No CUDA device found, skipping CUDA runtime test");
    return;
  }

  ActsExamples::CudaCandidateListContainer lists;
  lists.addSeed(7u, 3.0, 0.40, {10u, 11u, 12u});
  lists.addSeed(11u, 4.0, 0.50, {11u, 20u});

  ActsExamples::CudaStream stream;
  lists.moveToDevice(stream.get());
  stream.synchronize();

  BOOST_CHECK(lists.isOnDevice());
  BOOST_CHECK(lists.deviceArrays().seedHitIndex != nullptr);
  BOOST_CHECK(lists.deviceArrays().candidateOffsets != nullptr);
  BOOST_CHECK(lists.deviceArrays().candidateIndices != nullptr);
  BOOST_CHECK_EQUAL(lists.deviceArrays().nSeeds, 2u);
  BOOST_CHECK_EQUAL(lists.deviceArrays().nCandidates, 5u);

  const std::uint32_t seedHit{lists.seedHitIndex(0)};
  const double sector{lists.seedSector(1)};
  const double theta{lists.seedTheta(1)};
  const std::uint32_t sharedHit{lists.candidateHitIndex(0, 1)};

  // Overwrite the host columns, so the copy back is the only source of the
  // values checked below
  lists.setSeed(0, 0u, 0.0, 0.0);
  lists.setSeed(1, 0u, 0.0, 0.0);
  for (std::size_t i = 0; i < lists.nCandidates(); ++i) {
    lists.setCandidate(i, 0u);
  }

  lists.moveToHost(stream.get());
  stream.synchronize();

  BOOST_CHECK_EQUAL(lists.nSeeds(), 2u);
  BOOST_CHECK_EQUAL(lists.nCandidates(), 5u);
  BOOST_CHECK_EQUAL(lists.seedHitIndex(0), seedHit);
  BOOST_CHECK_EQUAL(lists.seedSector(1), sector);
  BOOST_CHECK_EQUAL(lists.seedTheta(1), theta);
  BOOST_CHECK_EQUAL(lists.candidateBegin(1), 3u);
  BOOST_CHECK_EQUAL(lists.candidateEnd(1), 5u);
  BOOST_CHECK_EQUAL(lists.candidateHitIndex(0, 1), sharedHit);
  BOOST_CHECK_EQUAL(lists.candidateHitIndex(1, 0), sharedHit);
  BOOST_CHECK_EQUAL(lists.candidateHitIndex(1, 1), 20u);

  lists.clearDevice();

  BOOST_CHECK(!lists.isOnDevice());
  BOOST_CHECK(lists.deviceArrays().candidateIndices == nullptr);
}

BOOST_AUTO_TEST_CASE(CudaCandidateListsFromTree) {
  using ActsExamples::GlobalPatternFinder_t;
  using StationName = MuonId::StationName;
  const auto gctx = GeometryContext::dangerouslyDefaultConstruct();
  const auto surface = Surface::makeShared<PlaneSurface>(
      Transform3::Identity(), std::make_shared<RectangleBounds>(1._m, 1._m));

  // Middle, outer and inner layer hits. The inner one is no seed
  ActsExamples::MuonSpacePointBucket bucket{};
  for (const StationName station :
       {StationName::BML, StationName::BOL, StationName::BIL, StationName::BML,
        StationName::BML}) {
    bucket.push_back(makeSpacePoint(MuonId::TechField::Rpc, false,
                                    Vector3{0., 10., 100.}, station));
  }
  std::vector<ActsExamples::HitPayload> payloads{};
  for (const auto& sp : bucket) {
    payloads.emplace_back(gctx, &sp, &bucket, Transform3::Identity(),
                          surface.get());
  }

  // Tree coordinates are sector and theta. Hit 0 enters a second sector, as a
  // hit close to the sector border does
  ActsExamples::SearchTree_t::vector_t entries{};
  auto add = [&](std::size_t hit, double sector, double theta) {
    entries.push_back({{sector, theta}, &payloads[hit]});
  };
  add(0, 4., 0.50);
  add(1, 4., 0.51);
  add(2, 4., 0.505);
  add(3, 4., 0.80);
  add(4, 6., 0.50);
  add(0, 5., 0.50);
  const ActsExamples::SearchTree_t tree{std::move(entries)};

  const ActsExamples::SeedSelector selector{
      ActsExamples::SeedSelector::Config{}};
  const auto lists =
      ActsExamples::makeCudaCandidateLists<GlobalPatternFinder_t>(
          tree, payloads, selector);

  // Every tree entry but the inner hit is a seed. The candidates are the hits
  // inside the sector and theta window of the seed
  BOOST_REQUIRE_EQUAL(lists.nSeeds(), 5u);
  for (std::size_t seed = 0; seed < lists.nSeeds(); ++seed) {
    auto candidates = lists.candidateHitIndices(seed);
    std::vector<std::uint32_t> rows{candidates.begin(), candidates.end()};
    std::ranges::sort(rows);

    std::vector<std::uint32_t> expected{};
    if (lists.seedSector(seed) == 4. && lists.seedTheta(seed) < 0.6) {
      expected = {0u, 1u, 2u};
    } else {
      expected = {lists.seedHitIndex(seed)};
    }
    BOOST_CHECK_EQUAL_COLLECTIONS(rows.begin(), rows.end(), expected.begin(),
                                  expected.end());
  }
}

BOOST_AUTO_TEST_CASE(CudaPatternRecordHostConstruction) {
  // Two seeds with two pattern slots each and three hit slots per pattern
  ActsExamples::CudaPatternRecordContainer patterns{2, 2, 3};

  BOOST_CHECK_EQUAL(patterns.nSeeds(), 2u);
  BOOST_CHECK_EQUAL(patterns.nSlots(), 4u);
  BOOST_CHECK_EQUAL(patterns.slot(1, 1), 3u);
  BOOST_CHECK(!patterns.isOnDevice());
  BOOST_CHECK_EQUAL(patterns.nPatterns(0), 0u);
  BOOST_CHECK(!patterns.overflow(0));
  BOOST_CHECK_EQUAL(patterns.pattern(0).lastInsertedHit,
                    ActsExamples::cudaInvalidHitPosition);

  ActsExamples::CudaPatternRecordRow row{};
  row.sector = 3;
  row.patTheta = 0.40;
  row.linePos = Vector3{1.0, 2.0, 3.0};
  row.leverArm = 40.0;
  row.nPrecisionLayers = 5u;
  row.nHits = 2u;
  row.lastInsertedHit = 1u;
  row.lineAnchorHit = 0u;
  row.flags = ActsExamples::CudaPatternFlags::isOverlap;
  row.status = ActsExamples::CudaPatternStatus::ok;

  const std::size_t slot{patterns.slot(0, 0)};
  patterns.setPattern(slot, row);
  patterns.setHit(slot, 0, 10u, 1u);
  patterns.setHit(slot, 1, 11u, 2u);
  patterns.setNPatterns(0, 1u);
  patterns.setOverflow(1, true);

  const ActsExamples::CudaPatternRecordRow back{patterns.pattern(slot)};
  BOOST_CHECK_EQUAL(back.sector, 3);
  BOOST_CHECK(back.linePos == row.linePos);
  BOOST_CHECK_EQUAL(back.lastInsertedHit, 1u);
  BOOST_CHECK_EQUAL(back.flags, row.flags);
  // A hit reference is a position in the pattern's own hit list, so it gives
  // the payload row and the global layer of the hit
  BOOST_CHECK_EQUAL(patterns.hitIndex(slot, back.lastInsertedHit), 11u);
  BOOST_CHECK_EQUAL(patterns.globLayer(slot, back.lastInsertedHit), 2u);
  BOOST_CHECK_EQUAL(patterns.hitIndices(slot).size(), 2u);
  BOOST_CHECK_EQUAL(patterns.nPatterns(0), 1u);
  BOOST_CHECK(patterns.overflow(1));

  BOOST_CHECK_THROW(patterns.slot(2, 0), std::out_of_range);
  BOOST_CHECK_THROW(patterns.slot(0, 2), std::out_of_range);
  BOOST_CHECK_THROW(patterns.setHit(slot, 3, 0u), std::out_of_range);
  BOOST_CHECK_THROW(patterns.setNPatterns(0, 3u), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(CudaPatternRecordDeviceRoundTrip) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    BOOST_TEST_MESSAGE("No CUDA device found, skipping CUDA runtime test");
    return;
  }

  ActsExamples::CudaPatternRecordContainer patterns{2, 2, 3};
  // A host value that must not reach the device: the records are output
  patterns.setNPatterns(0, 1u);

  ActsExamples::CudaStream stream;
  patterns.moveToDevice(stream.get());
  stream.synchronize();

  const ActsExamples::CudaPatternRecordArrays device{patterns.deviceArrays()};
  BOOST_CHECK(patterns.isOnDevice());
  BOOST_CHECK(device.hitIndex != nullptr);
  BOOST_CHECK_EQUAL(device.capacityPerSeed, 2u);
  BOOST_CHECK_EQUAL(device.slot(1, 1), 3u);
  BOOST_CHECK_EQUAL(device.hitSlot(3, 2), 11u);

  // Stand in for a kernel: seed 1 filled two slots, the first holding hit 9.
  // The writes go on the container's stream, which does not wait for the
  // default stream
  const std::uint32_t nFilled{2u};
  const std::uint32_t hit{9u};
  BOOST_REQUIRE(cudaMemcpyAsync(device.nPatterns + 1, &nFilled, sizeof(nFilled),
                                cudaMemcpyHostToDevice,
                                stream.get()) == cudaSuccess);
  BOOST_REQUIRE(cudaMemcpyAsync(device.hitIndex + device.hitSlot(2, 0), &hit,
                                sizeof(hit), cudaMemcpyHostToDevice,
                                stream.get()) == cudaSuccess);

  patterns.moveToHost(stream.get());
  stream.synchronize();

  BOOST_CHECK_EQUAL(patterns.nPatterns(0), 0u);
  BOOST_CHECK_EQUAL(patterns.nPatterns(1), 2u);
  BOOST_CHECK(!patterns.overflow(1));
  BOOST_CHECK_EQUAL(patterns.hitIndex(patterns.slot(1, 0), 0), 9u);
  BOOST_CHECK_EQUAL(patterns.pattern(0).status,
                    ActsExamples::CudaPatternStatus::empty);

  patterns.clearDevice();
  BOOST_CHECK(!patterns.isOnDevice());
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
