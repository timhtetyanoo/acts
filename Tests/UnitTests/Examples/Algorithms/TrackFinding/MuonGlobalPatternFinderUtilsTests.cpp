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
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "ActsExamples/EventData/MuonSpacePoint.hpp"
#include "ActsExamples/TrackFinding/MuonGlobalPatternFinderUtils.hpp"
#include "ActsTests/CommonHelpers/FloatComparisons.hpp"

#include <cstdint>
#include <tuple>
#include <vector>

using namespace ActsExamples;
using namespace Acts::UnitLiterals;
namespace Utils = ActsExamples::MuonGlobalPatternFinderUtils;

namespace ActsTests {

namespace {

using MuonId = MuonSpacePoint::MuonId;

MuonSpacePoint makeSpacePoint(MuonId::TechField tech, bool measEta,
                              bool measPhi, double z = 0.) {
  MuonId id{};
  id.setChamber(MuonId::StationName::BIL, MuonId::DetSide::A, 1u, tech);
  id.setCoordFlags(measEta, measPhi);
  MuonSpacePoint sp{};
  sp.setId(id);
  sp.defineCoordinates(Acts::Vector3{0., 0., z},
                       Acts::Vector3{Acts::Vector3::UnitX()},
                       Acts::Vector3{Acts::Vector3::UnitY()});
  return sp;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(MuonGlobalPatternFinderUtilsSuite)

BOOST_AUTO_TEST_CASE(PrecisionHit) {
  using enum MuonId::TechField;
  const std::vector<std::tuple<MuonId::TechField, bool, bool, bool>> cases{
      // technology, measuresEta, measuresPhi, isPrecision
      {Mdt, true, false, true},  {Mm, true, false, true},
      {sTgc, true, false, true},  // strips
      {sTgc, true, true, false},  // pads
      {sTgc, false, true, false},  // wires
      {Rpc, true, false, false}, {Rpc, true, true, false},
      {Tgc, true, false, false}, {Tgc, false, true, false},
  };
  for (const auto& [tech, measEta, measPhi, expected] : cases) {
    BOOST_CHECK_EQUAL(
        Utils::isPrecisionHit(makeSpacePoint(tech, measEta, measPhi)),
        expected);
  }
}

BOOST_AUTO_TEST_CASE(ChamberId) {
  const auto sensitiveId = Acts::GeometryIdentifier{}
                               .withVolume(3)
                               .withLayer(2)
                               .withSensitive(5);
  const auto chamberId =
      Acts::GeometryIdentifier{}.withVolume(3).withLayer(2);
  BOOST_CHECK_EQUAL(Utils::toChamberId(sensitiveId), chamberId);
  BOOST_CHECK_EQUAL(Utils::toChamberId(chamberId), chamberId);
}

BOOST_AUTO_TEST_CASE(SpacePointFrameToGlobal) {
  const Acts::Transform3 volumeLocToGlob =
      Acts::Translation3{Acts::Vector3{4._m, -1._m, 7._m}} *
      Acts::AngleAxis3{30._degree, Acts::Vector3::UnitZ()} *
      Acts::AngleAxis3{-50._degree, Acts::Vector3::UnitX()};

  /// Replicate the space point frame of the MuonSpacePointDigitizer
  const Acts::Transform3 globToSpFrame =
      Acts::AngleAxis3{90._degree, Acts::Vector3::UnitZ()} *
      volumeLocToGlob.inverse();

  const Acts::Transform3 spFrameToGlob =
      Utils::spacePointFrameToGlobal(volumeLocToGlob);

  const std::vector<Acts::Vector3> globalPoints{
      Acts::Vector3::Zero(), Acts::Vector3{4._m, -1._m, 7._m},
      Acts::Vector3{-2._m, 5._m, 11._m}, Acts::Vector3{7._m, 7._m, -3._m}};
  for (const Acts::Vector3& glob : globalPoints) {
    CHECK_CLOSE_ABS(spFrameToGlob * (globToSpFrame * glob), glob, 1e-9);
  }

  const Acts::Vector3 globDir = Acts::Vector3{1., -2., 0.5}.normalized();
  CHECK_CLOSE_ABS(spFrameToGlob.linear() * (globToSpFrame.linear() * globDir),
                  globDir, 1e-12);
}

BOOST_AUTO_TEST_CASE(LayerNumbers) {
  using enum MuonId::TechField;
  BOOST_CHECK(Utils::layerNumbers(MuonSpacePointBucket{}).empty());

  /// Unsorted bucket with two space points per layer, where one of them
  /// is displaced by less than the tolerance
  const MuonSpacePointBucket bucket{
      makeSpacePoint(Mdt, true, false, 26._mm),
      makeSpacePoint(Mdt, true, false, 0._mm),
      makeSpacePoint(Rpc, true, true, 300._mm),
      makeSpacePoint(Mdt, true, false, 1._um),
      makeSpacePoint(Mdt, true, false, 26._mm),
      makeSpacePoint(Rpc, false, true, 310._mm),
  };
  const std::vector<std::uint8_t> expected{1, 0, 2, 0, 1, 3};
  const std::vector<std::uint8_t> layers = Utils::layerNumbers(bucket);
  BOOST_CHECK_EQUAL_COLLECTIONS(layers.begin(), layers.end(), expected.begin(),
                                expected.end());

  /// A tolerance larger than the RPC gas gap separation merges both layers
  const std::vector<std::uint8_t> expectedLoose{1, 0, 2, 0, 1, 2};
  const std::vector<std::uint8_t> layersLoose =
      Utils::layerNumbers(bucket, 20._mm);
  BOOST_CHECK_EQUAL_COLLECTIONS(layersLoose.begin(), layersLoose.end(),
                                expectedLoose.begin(), expectedLoose.end());
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
