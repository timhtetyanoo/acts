// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <boost/test/unit_test.hpp>

#include "Acts/Definitions/Algebra.hpp"
#include "ActsExamples/EventData/CudaHitPayload.hpp"
#include "ActsExamples/EventData/MuonSpacePoint.hpp"
#include "ActsExamples/Utilities/CudaStream.hpp"

#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>

namespace ActsTests {

using namespace Acts;

using MuonId = ActsExamples::MuonSpacePoint::MuonId;

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
  BOOST_CHECK_EQUAL(ActsExamples::stationKey(reference),
                    ActsExamples::stationKey(
                        makeId(StationName::BIS, DetSide::A, 3)));

  // Each of the three station fields separates the keys
  BOOST_CHECK_NE(ActsExamples::stationKey(reference),
                 ActsExamples::stationKey(
                     makeId(StationName::BIL, DetSide::A, 3)));
  BOOST_CHECK_NE(ActsExamples::stationKey(reference),
                 ActsExamples::stationKey(
                     makeId(StationName::BIS, DetSide::C, 3)));
  BOOST_CHECK_NE(ActsExamples::stationKey(reference),
                 ActsExamples::stationKey(
                     makeId(StationName::BIS, DetSide::A, 4)));

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
                  ActsExamples::stationKey(makeId(
                      MuonId::StationName::BIS, MuonId::DetSide::A, 3)),
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

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
