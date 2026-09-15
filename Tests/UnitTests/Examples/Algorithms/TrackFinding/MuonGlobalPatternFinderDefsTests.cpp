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
#include "ActsExamples/EventData/MuonGlobalPatternFinderEvent.hpp"
#include "ActsExamples/EventData/MuonSpacePoint.hpp"
#include "ActsExamples/TrackFinding/MuonGlobalPatternFinderDefs.hpp"
#include "ActsTests/CommonHelpers/FloatComparisons.hpp"

#include <cmath>
#include <numbers>
#include <sstream>
#include <stdexcept>

using namespace ActsExamples;
using namespace ActsExamples::MuonGlobalPatternFinderDefs;
using namespace Acts::UnitLiterals;

namespace ActsTests {

namespace {

using MuonId = MuonSpacePoint::MuonId;

/// All hits are placed at global (0, R, 0), where the azimuthal gradient
/// is (-1/R, 0, 0)
constexpr double R = 5._m;
constexpr double covPhi = 4.;
constexpr double covEta = 9.;

const Acts::Transform3 localToGlobal{
    Acts::Translation3{Acts::Vector3{0., R, 0.}}};

MuonSpacePoint makeSpacePoint(MuonId::TechField tech, bool measEta,
                              bool measPhi, const Acts::Vector3& sensorDir,
                              const Acts::Vector3& toNext) {
  MuonId id{};
  id.setChamber(MuonId::StationName::BML, MuonId::DetSide::A, 3u, tech);
  id.setCoordFlags(measEta, measPhi);
  MuonSpacePoint sp{};
  sp.setId(id);
  sp.defineCoordinates(Acts::Vector3::Zero(), Acts::Vector3{sensorDir},
                       Acts::Vector3{toNext});
  sp.setCovariance(covPhi, covEta, 0.);
  return sp;
}

HitPayload makePayload(const MuonSpacePoint& sp) {
  return HitPayload{&sp, nullptr, localToGlobal, 2u, MuonStationIndex::BM};
}

}  // namespace

BOOST_AUTO_TEST_SUITE(MuonGlobalPatternFinderDefsSuite)

BOOST_AUTO_TEST_CASE(PayloadBasics) {
  using enum MuonId::TechField;
  const MuonSpacePoint sp = makeSpacePoint(Rpc, true, false, Acts::Vector3::UnitX(),
                                           Acts::Vector3::UnitY());
  const MuonSpacePointBucket bucket{sp};
  const HitPayload hit{&bucket.front(), &bucket, localToGlobal, 2u,
                       MuonStationIndex::BM};
  CHECK_CLOSE_ABS(hit.position, Acts::Vector3(0., R, 0.), 1e-12);
  BOOST_CHECK_EQUAL(hit.sp, &bucket.front());
  BOOST_CHECK_EQUAL(hit.bucket, &bucket);
  BOOST_CHECK_EQUAL(hit.station, MuonStationIndex::BM);
  BOOST_CHECK_EQUAL(static_cast<int>(hit.locLayer), 2);
  BOOST_CHECK(!hit.isStraw);
  BOOST_CHECK(!hit.isPrecision);
  BOOST_CHECK(hit.measuresEta);
  BOOST_CHECK(!hit.measuresPhi);
  BOOST_CHECK(!hit.nonOrthogonalStrips);

  const HitPayload sameHit = makePayload(bucket.front());
  const HitPayload otherHit = makePayload(sp);
  BOOST_CHECK(hit == sameHit);
  BOOST_CHECK(!(hit == otherHit));

  const CandidateHit c1{&hit, hit.station, 4u};
  const CandidateHit c2{&sameHit, sameHit.station, 1u};
  const CandidateHit c3{&otherHit, otherHit.station, 4u};
  BOOST_CHECK(c1 == c2);
  BOOST_CHECK(!(c1 == c3));
  BOOST_CHECK(c1 == sameHit);
  BOOST_CHECK_EQUAL(c1.sp(), hit.sp);

  std::ostringstream oss{};
  oss << c1;
  BOOST_CHECK(!oss.str().empty());
}

BOOST_AUTO_TEST_CASE(OrthogonalStrips) {
  using enum MuonId::TechField;
  /// Eta-only strip
  const MuonSpacePoint etaSp = makeSpacePoint(
      Rpc, true, false, Acts::Vector3::UnitX(), Acts::Vector3::UnitY());
  const HitPayload etaHit = makePayload(etaSp);
  CHECK_SMALL(etaHit.phiCov, 1e-15);
  CHECK_CLOSE_ABS(etaHit.sensorDir, Acts::Vector3::UnitX(), 1e-12);
  CHECK_CLOSE_ABS(etaHit.secondaryMeasDir, Acts::Vector3::UnitY(), 1e-12);
  CHECK_CLOSE_REL(etaHit.residualVariance(Acts::Vector3::UnitY(), false),
                  covEta, 1e-12);
  CHECK_CLOSE_REL(etaHit.residualVariance(
                      Acts::Vector3{1., 1., 0.}.normalized(), false),
                  0.5 * covEta, 1e-12);

  /// Eta & phi strip: phiCov = covPhi * (sensorDir . gradPhi)^2
  const MuonSpacePoint combSp = makeSpacePoint(
      Rpc, true, true, Acts::Vector3::UnitX(), Acts::Vector3::UnitY());
  const HitPayload combHit = makePayload(combSp);
  BOOST_CHECK(!combHit.nonOrthogonalStrips);
  CHECK_CLOSE_REL(combHit.phiCov, covPhi / (R * R), 1e-12);
  const Acts::Vector3 diag = Acts::Vector3{1., 1., 0.}.normalized();
  CHECK_CLOSE_REL(combHit.residualVariance(diag, false),
                  0.5 * covEta + 0.5 * covPhi, 1e-12);
}

BOOST_AUTO_TEST_CASE(NonOrthogonalStrips) {
  using enum MuonId::TechField;
  /// Stereo strips rotated by 60 degrees w.r.t. the next-sensor direction
  const Acts::Vector3 stereoDir{std::cos(60._degree), std::sin(60._degree),
                                0.};
  const MuonSpacePoint sp =
      makeSpacePoint(sTgc, true, true, stereoDir, Acts::Vector3::UnitY());
  const HitPayload hit = makePayload(sp);
  BOOST_CHECK(hit.nonOrthogonalStrips);
  BOOST_CHECK(!hit.isPrecision);
  /// The sensor direction is orthogonal to the next sensor in the plane
  CHECK_CLOSE_ABS(std::abs(hit.sensorDir.x()), 1., 1e-12);
  CHECK_CLOSE_ABS(hit.secondaryMeasDir, stereoDir, 1e-12);

  /// phiCov = covEta * (toNext . grad)^2 + covPhi * (stereoDir . grad)^2
  const double stereoGrad = stereoDir.x() / R;
  CHECK_CLOSE_REL(hit.phiCov, covPhi * stereoGrad * stereoGrad, 1e-12);

  /// The primary measurement direction is the next-sensor direction
  CHECK_CLOSE_REL(hit.residualVariance(Acts::Vector3::UnitY(), false),
                  covEta + covPhi * 0.75, 1e-12);
}

BOOST_AUTO_TEST_CASE(PhiOnlyStrip) {
  using enum MuonId::TechField;
  const MuonSpacePoint sp = makeSpacePoint(
      Tgc, false, true, Acts::Vector3::UnitY(), Acts::Vector3::UnitX());
  const HitPayload hit = makePayload(sp);
  CHECK_CLOSE_REL(hit.phiCov, covPhi / (R * R), 1e-12);
  CHECK_CLOSE_ABS(hit.sensorDir, Acts::Vector3::Zero(), 1e-12);
  BOOST_CHECK_THROW(hit.residualVariance(Acts::Vector3::UnitY(), false),
                    std::runtime_error);
}

BOOST_AUTO_TEST_CASE(Straws) {
  using enum MuonId::TechField;
  constexpr double driftR = 7._mm;
  const double discCov = driftR * driftR + covEta;
  const Acts::Vector3 diag = Acts::Vector3{1., 1., 0.}.normalized();

  /// Eta-only straw
  MuonSpacePoint sp = makeSpacePoint(Mdt, true, false, Acts::Vector3::UnitX(),
                                     Acts::Vector3::UnitY());
  sp.setRadius(driftR);
  const HitPayload hit = makePayload(sp);
  BOOST_CHECK(hit.isStraw);
  BOOST_CHECK(hit.isPrecision);
  CHECK_SMALL(hit.phiCov, 1e-15);
  CHECK_CLOSE_REL(hit.secondaryMeasDir.x(), discCov, 1e-12);
  CHECK_CLOSE_REL(hit.residualVariance(Acts::Vector3::UnitY(), false), discCov,
                  1e-12);
  CHECK_SMALL(hit.residualVariance(Acts::Vector3::UnitX(), false), 1e-12);
  CHECK_CLOSE_REL(hit.residualVariance(diag, false), 0.5 * discCov, 1e-12);
  CHECK_CLOSE_REL(hit.residualVariance(Acts::Vector3{1., 2., 0.}, true),
                  5. * discCov, 1e-12);

  /// Straw combined with a phi measurement:
  /// phiCov = discCov / R^2 + (sensorDir . grad)^2 * (covPhi - discCov)
  ///        = covPhi / R^2 for a sensor along the gradient
  MuonSpacePoint combSp = makeSpacePoint(
      Mdt, true, true, Acts::Vector3::UnitX(), Acts::Vector3::UnitY());
  combSp.setRadius(driftR);
  const HitPayload combHit = makePayload(combSp);
  BOOST_CHECK(!combHit.nonOrthogonalStrips);
  CHECK_CLOSE_REL(combHit.phiCov, covPhi / (R * R), 1e-12);
  CHECK_CLOSE_REL(combHit.residualVariance(diag, false),
                  0.5 * discCov + 0.5 * covPhi, 1e-12);
  /// The phi measurement is cancelled by the projection
  CHECK_CLOSE_REL(combHit.residualVariance(Acts::Vector3{1., 2., 0.}, true),
                  5. * discCov, 1e-12);
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
