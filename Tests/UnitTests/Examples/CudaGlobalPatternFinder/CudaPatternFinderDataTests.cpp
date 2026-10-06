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
ActsExamples::MuonSpacePoint makeSpacePoint(MuonId::TechField tech,
                                            bool measPhi, const Vector3& pos) {
  MuonId id{};
  id.setChamber(MuonId::StationName::BML, MuonId::DetSide::A, 3u, tech);
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

/// @brief The intrinsic variance along a direction, from the columns of a row
double varianceFromRow(const ActsExamples::CudaHitPayloadRow& row,
                       const Vector3& direction) {
  const bool measuresPhi{(row.flags & CudaHitFlags::measuresPhi) != 0u};
  if ((row.flags & CudaHitFlags::straw) != 0u) {
    const double dotProdSq{square(row.sensorDirection.dot(direction))};
    const double etaTerm{row.driftCovariance *
                         (direction.squaredNorm() - dotProdSq)};
    return measuresPhi ? etaTerm + dotProdSq * row.phiCovariance : etaTerm;
  }
  const double etaTerm{row.etaCovariance *
                       square(row.etaMeasurementDirection.dot(direction))};
  return measuresPhi
             ? etaTerm + row.phiCovariance *
                             square(row.phiMeasurementDirection.dot(direction))
             : etaTerm;
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
  row.phiVariance = 1.e-4;
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
  BOOST_CHECK_EQUAL(back.phiVariance, row.phiVariance);
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
  BOOST_CHECK_EQUAL(stripRow.phiVariance, strip.phiVariance(gctx));

  // The rows hold what is needed to evaluate the variance of the CPU hits
  const std::array<Vector3, 2> directions{Vector3{1., 0., 0.},
                                          Vector3{1., 2., 3.}.normalized()};
  for (const Vector3& direction : directions) {
    BOOST_CHECK_CLOSE_FRACTION(varianceFromRow(strawRow, direction),
                               straw.intrinsicVariance(gctx, direction),
                               1.e-12);
    BOOST_CHECK_CLOSE_FRACTION(varianceFromRow(stripRow, direction),
                               strip.intrinsicVariance(gctx, direction),
                               1.e-12);
  }
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

BOOST_AUTO_TEST_CASE(CudaPatternRecordHostConstruction) {
  ActsExamples::CudaPatternRecordContainer patterns{2, 3};

  BOOST_CHECK_EQUAL(patterns.nPatterns(), 2u);
  BOOST_CHECK_EQUAL(patterns.maxHitsPerPattern(), 3u);
  BOOST_CHECK(!patterns.empty());
  BOOST_CHECK(!patterns.isOnDevice());
  BOOST_CHECK_EQUAL(patterns.nHits(0), 0u);
  BOOST_CHECK_EQUAL(patterns.pattern(0).lastInsertedHit,
                    ActsExamples::cudaInvalidHitIndex);
  BOOST_CHECK_EQUAL(patterns.pattern(0).status,
                    ActsExamples::CudaPatternStatus::empty);

  ActsExamples::CudaPatternRecordRow row{};
  row.seedIndex = 4u;
  row.sector = 3;
  row.patPhi = 0.12;
  row.patTheta = 0.40;
  row.patPhiCov = 1.e-4;
  row.meanNormResidual2 = 0.05;
  row.lastResidual = 2.0;
  row.lastResSigma = 0.5;
  row.linePos = Vector3{1.0, 2.0, 3.0};
  row.lineDir = Vector3{0.0, 0.0, 1.0};
  row.leverArm = 40.0;
  row.bendPlaneNorm = Vector3{0.0, 1.0, 0.0};
  row.nPrecisionLayers = 5u;
  row.nTriggerLayers = 2u;
  row.nPhiLayers = 1u;
  row.nHits = 2u;
  row.lastInsertedHit = 11u;
  row.prevLayerHit = 10u;
  row.lineAnchorHit = 10u;
  row.flags = ActsExamples::CudaPatternFlags::needLineUpdate;
  row.status = ActsExamples::CudaPatternStatus::ok;

  patterns.setPattern(0, row);
  patterns.setHit(0, 0, 10u, 1u);
  patterns.setHit(0, 1, 11u, 2u);

  const ActsExamples::CudaPatternRecordRow back{patterns.pattern(0)};
  BOOST_CHECK_EQUAL(back.seedIndex, 4u);
  BOOST_CHECK_EQUAL(back.sector, 3);
  BOOST_CHECK_EQUAL(back.patTheta, 0.40);
  BOOST_CHECK(back.linePos == row.linePos);
  BOOST_CHECK(back.lineDir == row.lineDir);
  BOOST_CHECK(back.bendPlaneNorm == row.bendPlaneNorm);
  BOOST_CHECK_EQUAL(back.nPrecisionLayers, 5u);
  BOOST_CHECK_EQUAL(back.lastInsertedHit, 11u);
  BOOST_CHECK_EQUAL(back.flags, row.flags);
  BOOST_CHECK_EQUAL(back.status, ActsExamples::CudaPatternStatus::ok);
  BOOST_CHECK_EQUAL(patterns.nHits(0), 2u);
  BOOST_CHECK_EQUAL(patterns.hitIndex(0, 1), 11u);
  BOOST_CHECK_EQUAL(patterns.globLayer(0, 1), 2u);
  BOOST_CHECK_EQUAL(patterns.hitIndices(0).size(), 2u);
  BOOST_CHECK_EQUAL(patterns.hitIndices(0)[0], 10u);

  BOOST_CHECK_EQUAL(patterns.nHits(1), 0u);
  BOOST_CHECK_EQUAL(patterns.pattern(1).status,
                    ActsExamples::CudaPatternStatus::empty);

  BOOST_CHECK_THROW(patterns.pattern(2), std::out_of_range);
  BOOST_CHECK_THROW(patterns.setHit(0, 3, 0u), std::out_of_range);
  BOOST_CHECK_THROW(patterns.setNHits(0, 4u), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(CudaPatternRecordDeviceRoundTrip) {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    BOOST_TEST_MESSAGE("No CUDA device found, skipping CUDA runtime test");
    return;
  }

  ActsExamples::CudaPatternRecordContainer patterns{2, 3};

  ActsExamples::CudaPatternRecordRow row{};
  row.seedIndex = 4u;
  row.sector = -2;
  row.patTheta = 0.40;
  row.linePos = Vector3{1.0, 2.0, 3.0};
  row.leverArm = 40.0;
  row.nHits = 2u;
  row.lastInsertedHit = 11u;
  row.flags = ActsExamples::CudaPatternFlags::useBeamspot;
  row.status = ActsExamples::CudaPatternStatus::overflow;

  patterns.setPattern(0, row);
  patterns.setHit(0, 0, 10u, 1u);
  patterns.setHit(0, 1, 11u, 2u);

  ActsExamples::CudaStream stream;
  patterns.moveToDevice(stream.get());
  stream.synchronize();

  BOOST_CHECK(patterns.isOnDevice());
  BOOST_CHECK(patterns.deviceArrays().patTheta != nullptr);
  BOOST_CHECK(patterns.deviceArrays().hitIndex != nullptr);
  BOOST_CHECK_EQUAL(patterns.deviceArrays().nPatterns, 2u);
  BOOST_CHECK_EQUAL(patterns.deviceArrays().maxHitsPerPattern, 3u);
  BOOST_CHECK_EQUAL(patterns.deviceArrays().hitSlot(1, 2), 5u);

  // Overwrite the host columns, so the copy back is the only source of the
  // values checked below
  patterns.setPattern(0, ActsExamples::CudaPatternRecordRow{});
  patterns.setHit(0, 0, 0u, 0u);
  patterns.setHit(0, 1, 0u, 0u);

  patterns.moveToHost(stream.get());
  stream.synchronize();

  const ActsExamples::CudaPatternRecordRow back{patterns.pattern(0)};
  BOOST_CHECK_EQUAL(back.seedIndex, 4u);
  BOOST_CHECK_EQUAL(back.sector, -2);
  BOOST_CHECK_EQUAL(back.patTheta, 0.40);
  BOOST_CHECK(back.linePos == row.linePos);
  BOOST_CHECK_EQUAL(back.leverArm, 40.0);
  BOOST_CHECK_EQUAL(back.nHits, 2u);
  BOOST_CHECK_EQUAL(back.lastInsertedHit, 11u);
  BOOST_CHECK_EQUAL(back.flags, row.flags);
  BOOST_CHECK_EQUAL(back.status, ActsExamples::CudaPatternStatus::overflow);
  BOOST_CHECK_EQUAL(patterns.hitIndex(0, 0), 10u);
  BOOST_CHECK_EQUAL(patterns.hitIndex(0, 1), 11u);
  BOOST_CHECK_EQUAL(patterns.globLayer(0, 1), 2u);
  BOOST_CHECK_EQUAL(patterns.nHits(1), 0u);
  BOOST_CHECK_EQUAL(patterns.pattern(1).lastInsertedHit,
                    ActsExamples::cudaInvalidHitIndex);

  patterns.clearDevice();

  BOOST_CHECK(!patterns.isOnDevice());
  BOOST_CHECK(patterns.deviceArrays().hitIndex == nullptr);
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
