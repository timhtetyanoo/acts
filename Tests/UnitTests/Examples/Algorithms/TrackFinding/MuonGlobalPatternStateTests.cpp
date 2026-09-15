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
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Utilities/MathHelpers.hpp"
#include "Acts/Utilities/UnitVectors.hpp"
#include "Acts/Utilities/VectorHelpers.hpp"
#include "Acts/Utilities/detail/periodic.hpp"
#include "ActsExamples/EventData/MuonGlobalPatternFinderEvent.hpp"
#include "ActsExamples/EventData/MuonSpacePoint.hpp"
#include "ActsExamples/TrackFinding/MuonGlobalPatternFinderDefs.hpp"
#include "ActsTests/CommonHelpers/FloatComparisons.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <type_traits>

using namespace ActsExamples;
using namespace ActsExamples::MuonGlobalPatternFinderDefs;
using namespace Acts::UnitLiterals;

namespace ActsTests {

namespace {

using MuonId = MuonSpacePoint::MuonId;
using StationName = MuonId::StationName;
using TechField = MuonId::TechField;
using SectorProjector = MuonExpandedSector::SectorProjector;

static_assert(std::is_nothrow_move_constructible_v<PatternState>);
static_assert(std::is_nothrow_move_assignable_v<PatternState>);
static_assert(std::is_copy_constructible_v<PatternState>);
static_assert(std::is_copy_assignable_v<PatternState>);

constexpr double covPhi = 4.;
constexpr double covEta = 9.;
constexpr double R = 7._m;
/// Azimuthal variance of a phi hit at radius R whose phi measurement
/// direction is tangential
const double hitPhiCov = covPhi / (R * R);

const PatternFinderConfig cfg{};
const auto logger = Acts::getDefaultLogger("PatternStateTests",
                                           Acts::Logging::INFO);

/// @brief Unit vector tangential to a circle around the beam axis
Acts::Vector3 tangent(double phi) {
  return Acts::Vector3{-std::sin(phi), std::cos(phi), 0.};
}
/// @brief Global position at the given phi, radius & z
Acts::Vector3 atPhi(double phi, double r = R, double z = 3._m) {
  return Acts::Vector3{r * std::cos(phi), r * std::sin(phi), z};
}
/// @brief Raw expanded sector number
std::int8_t expSector(unsigned msSector,
                      SectorProjector proj = SectorProjector::center) {
  return MuonExpandedSector{msSector, proj}.sector();
}

/// @brief Owns the space points & hit payloads with stable addresses. The
///        space points are defined directly in the global frame.
class HitFactory {
 public:
  /// @brief Open a new bucket to which the subsequent hits are added
  void newBucket() { m_buckets.emplace_back().reserve(s_bucketCapacity); }

  const HitPayload& add(TechField tech, StationName st, bool measEta,
                        bool measPhi, const Acts::Vector3& pos,
                        const Acts::Vector3& sensorDir,
                        const Acts::Vector3& toNext) {
    MuonId id{};
    id.setChamber(st, MuonId::DetSide::A, 1u, tech);
    id.setCoordFlags(measEta, measPhi);
    MuonSpacePoint sp{};
    sp.setId(id);
    sp.defineCoordinates(Acts::Vector3{pos}, Acts::Vector3{sensorDir},
                         Acts::Vector3{toNext});
    sp.setCovariance(covPhi, covEta, 0.);

    if (m_buckets.empty() || m_buckets.back().size() == s_bucketCapacity) {
      newBucket();
    }
    MuonSpacePointBucket& bucket = m_buckets.back();
    bucket.push_back(std::move(sp));
    return m_payloads.emplace_back(&bucket.back(), &bucket, s_identity,
                                   std::uint8_t{0u}, stationIndex(st));
  }
  /// @brief Eta-only precision hit with the sensor along phi
  const HitPayload& eta(StationName st, const Acts::Vector3& pos,
                        const Acts::Vector3& sensorDir) {
    return add(TechField::Mdt, st, true, false, pos, sensorDir,
               Acts::Vector3::UnitZ());
  }
  const HitPayload& eta(StationName st, double phi) {
    return eta(st, atPhi(phi), tangent(phi));
  }
  /// @brief Trigger hit measuring eta & phi at radius R
  const HitPayload& etaPhi(StationName st, double phi) {
    return add(TechField::Rpc, st, true, true, atPhi(phi), tangent(phi),
               Acts::Vector3::UnitZ());
  }
  /// @brief Phi-only trigger hit at radius R
  const HitPayload& phiOnly(StationName st, double phi) {
    return add(TechField::Tgc, st, false, true, atPhi(phi),
               Acts::Vector3::UnitZ(), tangent(phi));
  }
  const MuonSpacePointBucket& bucket(std::size_t idx) const {
    return m_buckets.at(idx);
  }

 private:
  static constexpr std::size_t s_bucketCapacity = 64;
  static inline const Acts::Transform3 s_identity{
      Acts::Transform3::Identity()};
  std::deque<MuonSpacePointBucket> m_buckets{};
  std::deque<HitPayload> m_payloads{};
};

CandidateHit candidate(const HitPayload& hit, std::uint8_t globLayer) {
  return CandidateHit{&hit, hit.station, globLayer};
}

}  // namespace

BOOST_AUTO_TEST_SUITE(MuonGlobalPatternStateSuite)

BOOST_AUTO_TEST_CASE(Construction) {
  HitFactory factory{};
  const HitPayload& seedHit = factory.eta(StationName::BML, 0.);
  const HitPayload& otherHit = factory.eta(StationName::BML, 0.01);
  const CandidateHit seed = candidate(seedHit, 3u);

  PatternState pat{seed, expSector(1u), &cfg, logger.get()};
  BOOST_CHECK_EQUAL(pat.expSect.msSector(), 1u);
  CHECK_CLOSE_ABS(pat.patTheta, Acts::VectorHelpers::theta(seedHit.position),
                  1e-12);
  BOOST_CHECK_EQUAL(static_cast<int>(pat.nPrecisionLayers), 1);
  BOOST_CHECK_EQUAL(static_cast<int>(pat.nTriggerLayers), 0);
  BOOST_CHECK_EQUAL(static_cast<int>(pat.nPhiLayers), 0);
  BOOST_CHECK_EQUAL(static_cast<int>(pat.nBendingLayers()), 1);
  BOOST_CHECK_EQUAL(static_cast<int>(pat.nStations(false)), 1);
  BOOST_CHECK_EQUAL(static_cast<int>(pat.nStations(true)), 0);
  BOOST_CHECK(pat.needLineUpdate);
  BOOST_CHECK(!pat.isFinalized);
  BOOST_CHECK(pat.lastInsertedHit == seed);
  BOOST_CHECK(pat.prevLayerHit == seed);
  BOOST_CHECK(pat.lineAnchorHit == seed);
  BOOST_CHECK_EQUAL(
      pat.hitsPerStation[stationIdx(MuonStationIndex::BM)].size(), 1u);
  BOOST_CHECK(pat.isInPattern(seedHit));
  BOOST_CHECK(!pat.isInPattern(otherHit));
  BOOST_CHECK_EQUAL(pat.getParentBuckets().size(), 1u);
  BOOST_CHECK_EQUAL(pat.getParentBuckets().front(), &factory.bucket(0));

  /// Without phi hits, the pattern phi is the sector centre with the variance
  /// of a uniform distribution over the sector
  CHECK_SMALL(pat.patPhi, 1e-12);
  CHECK_CLOSE_REL(pat.patPhiCov,
                  Acts::square(MuonSectorMapping::sectorSize(1)) / 3., 1e-12);
  CHECK_CLOSE_ABS(pat.bendPlaneNorm, Acts::Vector3::UnitY(), 1e-12);

  /// Trigger seed in the overlap between sector 1 & 2
  const double overlapPhi = MuonSectorMapping::sectorOverlapPhi(1, 2);
  const HitPayload& trigHit = factory.add(
      TechField::Rpc, StationName::BML, true, false, atPhi(overlapPhi),
      tangent(overlapPhi), Acts::Vector3::UnitZ());
  const PatternState overlapPat{candidate(trigHit, 0u),
                                expSector(1u, SectorProjector::rightOverlap),
                                &cfg, logger.get()};
  BOOST_CHECK_EQUAL(static_cast<int>(overlapPat.nTriggerLayers), 1);
  BOOST_CHECK_EQUAL(static_cast<int>(overlapPat.nPrecisionLayers), 0);
  CHECK_CLOSE_ABS(overlapPat.patPhi, overlapPhi, 1e-12);
  CHECK_CLOSE_REL(overlapPat.patPhiCov,
                  Acts::square(MuonSectorMapping::s_sectorOverlap) / 3., 1e-12);

  std::ostringstream oss{};
  oss << brief(pat) << "\n" << detailed(pat);
  BOOST_CHECK(!oss.str().empty());

  BOOST_CHECK_THROW(PatternState(seed, expSector(1u), nullptr, logger.get()),
                    std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(PhiEstimate) {
  HitFactory factory{};
  const double phi1 = std::numbers::pi - 0.01;
  const double phi2 = -std::numbers::pi + 0.01;
  const HitPayload& hit1 = factory.etaPhi(StationName::BML, phi1);
  const HitPayload& hit2 = factory.etaPhi(StationName::BML, phi2);
  const HitPayload& hit3 = factory.phiOnly(StationName::BML, std::numbers::pi);
  CHECK_CLOSE_REL(hit1.phiCov, hitPhiCov, 1e-9);
  CHECK_CLOSE_REL(hit3.phiCov, hitPhiCov, 1e-9);

  /// Sector 9 is centred at phi = pi
  PatternState pat{candidate(hit1, 0u), expSector(9u), &cfg, logger.get()};
  BOOST_CHECK_EQUAL(static_cast<int>(pat.nPhiLayers), 1);
  CHECK_CLOSE_ABS(pat.patPhi, phi1, 1e-12);
  CHECK_CLOSE_REL(pat.patPhiCov, hitPhiCov, 1e-9);

  /// The weighted circular mean across the +-pi seam
  pat.hitsPerStation[stationIdx(MuonStationIndex::BM)].push_back(
      candidate(hit2, 1u));
  ++pat.nPhiLayers;
  pat.updatePatternPhi();
  CHECK_SMALL(Acts::detail::radian_sym(pat.patPhi - std::numbers::pi), 1e-9);
  CHECK_CLOSE_REL(pat.patPhiCov, hitPhiCov / 2., 1e-9);
  CHECK_CLOSE_ABS(
      pat.bendPlaneNorm,
      Acts::makeDirectionFromPhiTheta(pat.patPhi + 90._degree, 90._degree),
      1e-12);

  /// Phi-only hits contribute as well
  pat.phiOnlyHits.push_back(hit3);
  ++pat.nPhiLayers;
  pat.updatePatternPhi();
  CHECK_SMALL(Acts::detail::radian_sym(pat.patPhi - std::numbers::pi), 1e-9);
  CHECK_CLOSE_REL(pat.patPhiCov, hitPhiCov / 3., 1e-9);

  /// A phi hit with a radial sensor has no sensitivity in phi
  const HitPayload& radialHit =
      factory.add(TechField::Rpc, StationName::BML, true, true, atPhi(0.),
                  Acts::Vector3::UnitX(), Acts::Vector3::UnitZ());
  CHECK_SMALL(radialHit.phiCov, 1e-15);
  BOOST_CHECK_THROW(
      PatternState(candidate(radialHit, 0u), expSector(1u), &cfg, logger.get()),
      std::runtime_error);
}

BOOST_AUTO_TEST_CASE(PhiCompatibility) {
  HitFactory factory{};
  /// Without phi hits, the test hit must be inside the pattern sector. The
  /// width of sector 1 (with overlap) is 0.7 * pi / 8 ~ 0.275 rad
  const PatternState centrePat{candidate(factory.eta(StationName::BML, 0.), 0u),
                               expSector(1u), &cfg, logger.get()};
  BOOST_CHECK(centrePat.isPhiCompatible(factory.eta(StationName::BML, 0.1)));
  BOOST_CHECK(centrePat.isPhiCompatible(factory.eta(StationName::BML, -0.25)));
  BOOST_CHECK(!centrePat.isPhiCompatible(factory.eta(StationName::BML, 0.5)));

  /// In the overlap region, the test hit must be inside both sectors
  const double overlapPhi = MuonSectorMapping::sectorOverlapPhi(1, 2);
  const PatternState overlapPat{
      candidate(factory.eta(StationName::BML, overlapPhi), 0u),
      expSector(1u, SectorProjector::rightOverlap), &cfg, logger.get()};
  BOOST_CHECK(
      overlapPat.isPhiCompatible(factory.eta(StationName::BML, overlapPhi)));
  /// Inside sector 1 only
  BOOST_CHECK(!overlapPat.isPhiCompatible(factory.eta(StationName::BML, 0.05)));
  /// Inside sector 2 only
  BOOST_CHECK(!overlapPat.isPhiCompatible(factory.eta(StationName::BML, 0.45)));

  /// With phi hits, the test hit must be within nPhiSigma
  const PatternState phiPat{candidate(factory.etaPhi(StationName::BML, 0.), 0u),
                            expSector(1u), &cfg, logger.get()};
  const double window = cfg.nPhiSigma * std::sqrt(2. * hitPhiCov);
  BOOST_CHECK(
      phiPat.isPhiCompatible(factory.etaPhi(StationName::BML, 0.9 * window)));
  BOOST_CHECK(
      phiPat.isPhiCompatible(factory.etaPhi(StationName::BML, -0.9 * window)));
  BOOST_CHECK(
      !phiPat.isPhiCompatible(factory.etaPhi(StationName::BML, 1.1 * window)));
  BOOST_CHECK(
      !phiPat.isPhiCompatible(factory.etaPhi(StationName::BML, -1.1 * window)));
}

BOOST_AUTO_TEST_CASE(Projection) {
  HitFactory factory{};
  /// The bending plane of sector 1 is the xz-plane
  const PatternState pat{candidate(factory.eta(StationName::BML, 0.), 0u),
                         expSector(1u), &cfg, logger.get()};

  const HitPayload& alongY = factory.eta(
      StationName::BML, Acts::Vector3{7._m, 500._mm, 3._m},
      Acts::Vector3::UnitY());
  CHECK_CLOSE_ABS(pat.projToPhiPlane(alongY), Acts::Vector3(7._m, 0., 3._m),
                  1e-9);

  /// The hit moves along the inclined sensor
  const HitPayload& inclined = factory.eta(
      StationName::BML, Acts::Vector3{7._m, 400._mm, 3._m},
      Acts::Vector3{0.6, 0.8, 0.});
  CHECK_CLOSE_ABS(pat.projToPhiPlane(inclined),
                  Acts::Vector3(7._m - 300._mm, 0., 3._m), 1e-9);
}

BOOST_AUTO_TEST_CASE(LineParameters) {
  HitFactory factory{};
  const HitPayload& bi = factory.eta(
      StationName::BIL, Acts::Vector3{5._m, 0., 2._m}, Acts::Vector3::UnitY());
  const HitPayload& bm = factory.eta(
      StationName::BML, Acts::Vector3{7._m, 0., 3._m}, Acts::Vector3::UnitY());
  const HitPayload& bmNear =
      factory.eta(StationName::BML, Acts::Vector3{7._m, 0., 3._m + 20._mm},
                  Acts::Vector3::UnitY());
  const HitPayload& bmFar =
      factory.eta(StationName::BML, Acts::Vector3{7._m, 0., 3._m + 300._mm},
                  Acts::Vector3::UnitY());

  PatternState pat{candidate(bi, 0u), expSector(1u), &cfg, logger.get()};

  /// Line through two hits in different stations
  pat.lastInsertedHit = candidate(bm, 4u);
  pat.updateLineParameters(Acts::Vector3::Zero());
  const Acts::Vector3 d = bm.position - bi.position;
  BOOST_CHECK(!pat.useBeamspot);
  BOOST_CHECK(!pat.needLineUpdate);
  CHECK_CLOSE_ABS(pat.linePos, bi.position, 1e-9);
  CHECK_CLOSE_ABS(pat.lineDir, d.normalized(), 1e-12);
  CHECK_CLOSE_REL(pat.leverArm, d.norm(), 1e-12);

  /// Without an update request, the line is kept
  pat.lastInsertedHit = candidate(bmFar, 5u);
  pat.updateLineParameters(Acts::Vector3::Zero());
  CHECK_CLOSE_ABS(pat.lineDir, d.normalized(), 1e-12);

  /// Close-by hits in the same station: the beamspot defines the line
  const Acts::Vector3 beamSpot{0., 0., 10._mm};
  pat.lineAnchorHit = candidate(bm, 4u);
  pat.lastInsertedHit = candidate(bmNear, 5u);
  pat.needLineUpdate = true;
  pat.updateLineParameters(beamSpot);
  BOOST_CHECK(pat.useBeamspot);
  CHECK_CLOSE_ABS(pat.linePos, beamSpot, 1e-9);
  CHECK_CLOSE_ABS(pat.lineDir, (bmNear.position - beamSpot).normalized(),
                  1e-12);
  CHECK_CLOSE_REL(pat.leverArm, (bmNear.position - beamSpot).norm(), 1e-12);

  /// Distant hits in the same station define the line
  pat.lastInsertedHit = candidate(bmFar, 6u);
  pat.needLineUpdate = true;
  pat.updateLineParameters(beamSpot);
  BOOST_CHECK(!pat.useBeamspot);
  CHECK_CLOSE_ABS(pat.linePos, bm.position, 1e-9);
  CHECK_CLOSE_REL(pat.leverArm, 300._mm, 1e-9);
}

BOOST_AUTO_TEST_CASE(LineAnchor) {
  HitFactory factory{};
  const HitPayload& bi = factory.eta(StationName::BIL, 0.);
  const HitPayload& bm = factory.eta(StationName::BML, 0.);
  const HitPayload& bo = factory.eta(StationName::BOL, 0.);

  PatternState pat{candidate(bo, 10u), expSector(1u), &cfg, logger.get()};
  /// Single station: the anchor is the last inserted hit
  pat.lastInsertedHit = candidate(bo, 11u);
  pat.moveLineAnchorHit(candidate(bi, 0u));
  BOOST_CHECK_EQUAL(static_cast<int>(pat.lineAnchorHit.globLayer), 11);

  /// BI on layers 0-2, BM on layers 5-6, BO on layers 10-11
  auto& biHits = pat.hitsPerStation[stationIdx(MuonStationIndex::BI)];
  auto& bmHits = pat.hitsPerStation[stationIdx(MuonStationIndex::BM)];
  auto& boHits = pat.hitsPerStation[stationIdx(MuonStationIndex::BO)];
  biHits = {candidate(bi, 0u), candidate(bi, 1u), candidate(bi, 2u)};
  bmHits = {candidate(bm, 5u), candidate(bm, 6u)};
  boHits.push_back(candidate(bo, 11u));

  const auto checkAnchor = [&pat](const CandidateHit& ref,
                                  MuonStationIndex expStation,
                                  int expLayer) {
    pat.moveLineAnchorHit(ref);
    BOOST_CHECK_EQUAL(pat.lineAnchorHit.station, expStation);
    BOOST_CHECK_EQUAL(static_cast<int>(pat.lineAnchorHit.globLayer), expLayer);
  };
  checkAnchor(candidate(bo, 11u), MuonStationIndex::BM, 6);
  checkAnchor(candidate(bi, 0u), MuonStationIndex::BM, 5);
  checkAnchor(candidate(bm, 6u), MuonStationIndex::BO, 10);
}

BOOST_AUTO_TEST_CASE(Counters) {
  HitFactory factory{};
  const HitPayload& bi = factory.eta(StationName::BIL, 0.);
  factory.newBucket();
  const HitPayload& bm = factory.etaPhi(StationName::BML, 0.);

  PatternState pat{candidate(bi, 0u), expSector(1u), &cfg, logger.get()};
  auto& biHits = pat.hitsPerStation[stationIdx(MuonStationIndex::BI)];
  auto& bmHits = pat.hitsPerStation[stationIdx(MuonStationIndex::BM)];
  for (std::uint8_t lay = 1u; lay < 4u; ++lay) {
    biHits.push_back(candidate(bi, lay));
  }
  bmHits = {candidate(bm, 4u), candidate(bm, 5u)};
  pat.nPrecisionLayers = 4u;
  pat.nTriggerLayers = 2u;

  BOOST_CHECK_EQUAL(static_cast<int>(pat.nStations(false)), 2);
  /// Only BI has at least minStationLayers hits
  BOOST_CHECK_EQUAL(static_cast<int>(pat.nStations(true)), 1);
  BOOST_CHECK_EQUAL(static_cast<int>(pat.nBendingLayers()), 6);

  const auto buckets = pat.getParentBuckets();
  BOOST_CHECK_EQUAL(buckets.size(), 2u);
  BOOST_CHECK(std::ranges::find(buckets, &factory.bucket(0)) != buckets.end());
  BOOST_CHECK(std::ranges::find(buckets, &factory.bucket(1)) != buckets.end());

  pat.meanNormResidual2 = 12.;
  CHECK_CLOSE_REL(pat.getMeanResidual2(), 2., 1e-12);
  pat.isFinalized = true;
  CHECK_CLOSE_REL(pat.getMeanResidual2(), 12., 1e-12);
}

BOOST_AUTO_TEST_CASE(LineResidual) {
  HitFactory factory{};
  /// All hits lie in the bending plane of sector 1 (xz-plane). Straw hits
  /// without drift radius have a disc covariance of covEta
  const double discCov = covEta;
  const double patPhiCov =
      Acts::square(MuonSectorMapping::sectorSize(1)) / 3.;
  const Acts::Vector3 posA{5._m, 0., 2._m};
  const Acts::Vector3 posB{7._m, 0., 2.8_m};
  const Acts::Vector3 posC{9._m, 0., 3.6_m + 10._mm};

  /// Line through the hits A & B in different stations
  const auto makePattern = [&](const Acts::Vector3& sensorDir,
                               const HitPayload*& testHit) {
    const HitPayload& a = factory.eta(StationName::BIL, posA, sensorDir);
    const HitPayload& b = factory.eta(StationName::BML, posB, sensorDir);
    testHit = &factory.eta(StationName::BOL, posC, sensorDir);
    PatternState pat{candidate(a, 0u), expSector(1u), &cfg, logger.get()};
    pat.hitsPerStation[stationIdx(MuonStationIndex::BM)].push_back(
        candidate(b, 1u));
    pat.lastInsertedHit = candidate(b, 1u);
    pat.updateLineParameters(Acts::Vector3::Zero());
    BOOST_REQUIRE(!pat.useBeamspot);
    return pat;
  };
  const Acts::Vector3 d = posB - posA;
  const double alpha = (posC - posA).dot(d) / d.squaredNorm();
  const Acts::Vector3 resDir = (posC - posA - alpha * d).normalized();
  /// The distance to the line is the z-offset scaled by cos(line angle)
  const double expResidual = 10._mm * d.x() / d.norm();
  const double sumPreFactors2 =
      Acts::square(alpha - 1.) + Acts::square(alpha) + 1.;

  /// Sensors along the bending plane normal: the projection does not depend
  /// on the plane angle
  const HitPayload* testHit{nullptr};
  const PatternState pat = makePattern(Acts::Vector3::UnitY(), testHit);
  const LineTestRes res = pat.computeLineResidual(candidate(*testHit, 2u));
  CHECK_CLOSE_REL(res.residual, expResidual, 1e-9);
  CHECK_CLOSE_REL(res.sigma, std::sqrt(discCov * sumPreFactors2), 1e-9);
  BOOST_CHECK(res.result == LineTestDecision::eRejectHit);

  /// Inclined sensors: the projection adds (1 + projFactor^2) to the hit
  /// covariance & the uncertainty of the plane angle
  const Acts::Vector3 inclined{0., 0.8, 0.6};
  const PatternState inclPat = makePattern(inclined, testHit);
  const double projFactor = inclined.dot(resDir) / inclined.y();
  const double phiDerivative =
      projFactor * ((alpha - 1.) * posA.x() - alpha * posB.x() + posC.x());
  const LineTestRes inclRes =
      inclPat.computeLineResidual(candidate(*testHit, 2u));
  CHECK_CLOSE_REL(inclRes.residual, expResidual, 1e-9);
  CHECK_CLOSE_REL(
      inclRes.sigma,
      std::sqrt(discCov * (1. + Acts::square(projFactor)) * sumPreFactors2 +
                Acts::square(phiDerivative) * patPhiCov),
      1e-9);

  /// A test hit on the line anchor has no residual direction
  const HitPayload& onAnchor =
      factory.eta(StationName::BOL, posA, Acts::Vector3::UnitY());
  const LineTestRes zeroRes = pat.computeLineResidual(candidate(onAnchor, 2u));
  BOOST_CHECK_EQUAL(zeroRes.residual, std::numeric_limits<double>::max());
  BOOST_CHECK_EQUAL(zeroRes.sigma, 0.);

  /// Close-by hits in the same station: the line starts at the beamspot
  const Acts::Vector3 posA2{5._m, 0., 2._m + 20._mm};
  const HitPayload& a =
      factory.eta(StationName::BIL, posA, Acts::Vector3::UnitY());
  const HitPayload& a2 =
      factory.eta(StationName::BIL, posA2, Acts::Vector3::UnitY());
  PatternState bsPat{candidate(a, 0u), expSector(1u), &cfg, logger.get()};
  bsPat.hitsPerStation[stationIdx(MuonStationIndex::BI)].push_back(
      candidate(a2, 1u));
  bsPat.lastInsertedHit = candidate(a2, 1u);
  bsPat.updateLineParameters(Acts::Vector3::Zero());
  BOOST_REQUIRE(bsPat.useBeamspot);

  const HitPayload& c =
      factory.eta(StationName::BOL, posC, Acts::Vector3::UnitY());
  const double bsAlpha = posC.dot(posA2) / posA2.squaredNorm();
  const Acts::Vector3 bsResVec = posC - bsAlpha * posA2;
  const double resZ2 = Acts::square(bsResVec.normalized().z());
  const double covS1 =
      cfg.beamSpotLength * resZ2 + cfg.beamSpotRadius * (1. - resZ2);
  const LineTestRes bsRes = bsPat.computeLineResidual(candidate(c, 2u));
  CHECK_CLOSE_REL(bsRes.residual, bsResVec.norm(), 1e-9);
  /// The beamspot term replaces the anchor hit term
  CHECK_CLOSE_REL(bsRes.sigma,
                  std::sqrt(Acts::square(bsAlpha - 1.) * covS1 +
                            discCov * (Acts::square(bsAlpha) + 1.)),
                  1e-9);
}

BOOST_AUTO_TEST_CASE(LineCompatibility) {
  HitFactory factory{};
  const Acts::Vector3 beamSpot{Acts::Vector3::Zero()};
  const Acts::Vector3 unitY{Acts::Vector3::UnitY()};
  /// Hits close to a straight line from the origin with z = 0.4 * x
  const HitPayload& a =
      factory.eta(StationName::BIL, Acts::Vector3{5._m, 0., 2._m}, unitY);
  const HitPayload& aSameLayer = factory.eta(
      StationName::BIL, Acts::Vector3{5._m, 0., 2._m + 1._mm}, unitY);
  const HitPayload& b = factory.eta(
      StationName::BML, Acts::Vector3{7._m, 0., 2.8_m + 5._mm}, unitY);
  const HitPayload& bBranch = factory.eta(
      StationName::BML, Acts::Vector3{7._m, 0., 2.8_m - 3._mm}, unitY);
  const HitPayload& phiOff = factory.etaPhi(StationName::BML, 0.5);
  /// The line A -> B extrapolates to z = 3.61 m at x = 9 m
  const HitPayload& cGood = factory.eta(
      StationName::BOL, Acts::Vector3{9._m, 0., 3.61_m + 2._mm}, unitY);
  const HitPayload& cFar = factory.eta(
      StationName::BOL, Acts::Vector3{9._m, 0., 3.61_m + 500._mm}, unitY);

  PatternState pat{candidate(a, 0u), expSector(1u), &cfg, logger.get()};

  /// Phi-incompatible hits are rejected before the residual computation
  const LineTestRes phiRes = pat.checkLineComp(candidate(phiOff, 1u), beamSpot);
  BOOST_CHECK(phiRes.result == LineTestDecision::eRejectHit);
  BOOST_CHECK_EQUAL(phiRes.sigma, 0.);

  /// No branching on the seed layer without hits on other layers
  BOOST_CHECK(pat.checkLineComp(candidate(aSameLayer, 0u), beamSpot).result ==
              LineTestDecision::eRejectHit);

  /// First hit on a new layer: the line goes from the beamspot through the seed
  const LineTestRes resB = pat.checkLineComp(candidate(b, 1u), beamSpot);
  BOOST_CHECK(pat.useBeamspot);
  BOOST_CHECK(resB.result == LineTestDecision::eAddHit);
  pat.addHit(candidate(b, 1u), resB.residual, resB.sigma);

  /// The same hit is not added twice
  BOOST_CHECK(pat.checkLineComp(candidate(b, 1u), beamSpot).result ==
              LineTestDecision::eRejectHit);
  /// A compatible hit on the last layer branches the pattern
  BOOST_CHECK(pat.checkLineComp(candidate(bBranch, 1u), beamSpot).result ==
              LineTestDecision::eBranchPattern);

  /// Hit in the next station: the line is drawn through the anchor & last hit
  const LineTestRes resC = pat.checkLineComp(candidate(cGood, 2u), beamSpot);
  BOOST_CHECK(!pat.useBeamspot);
  CHECK_CLOSE_ABS(pat.linePos, a.position, 1e-9);
  BOOST_CHECK(resC.result == LineTestDecision::eAddHit);
  BOOST_CHECK(pat.checkLineComp(candidate(cFar, 2u), beamSpot).result ==
              LineTestDecision::eRejectHit);

  /// The acceptance window is doubled for hits in a new station. Place a hit
  /// at a residual between 3 & 6 sigma
  const HitPayload& probe = factory.eta(
      StationName::BOL, Acts::Vector3{9._m, 0., 3.61_m + 36._mm}, unitY);
  const LineTestRes probeRes = pat.computeLineResidual(candidate(probe, 2u));
  BOOST_REQUIRE(probeRes.residual > cfg.nResidualSigma * probeRes.sigma);
  BOOST_REQUIRE(probeRes.residual < 2. * cfg.nResidualSigma * probeRes.sigma);
  BOOST_CHECK(pat.checkLineComp(candidate(probe, 2u), beamSpot).result ==
              LineTestDecision::eAddHit);
  /// The same position in the station of the last hit is rejected
  const HitPayload& sameStProbe = factory.eta(
      StationName::BML, Acts::Vector3{9._m, 0., 3.61_m + 36._mm}, unitY);
  BOOST_CHECK(pat.checkLineComp(candidate(sameStProbe, 2u), beamSpot).result ==
              LineTestDecision::eRejectHit);
}

BOOST_AUTO_TEST_CASE(HitUpdates) {
  HitFactory factory{};
  const HitPayload& seed = factory.eta(StationName::BIL, 0.);
  const HitPayload& rpcPhi = factory.etaPhi(StationName::BML, 0.);
  const HitPayload& rpcEta =
      factory.add(TechField::Rpc, StationName::BML, true, false, atPhi(0.),
                  tangent(0.), Acts::Vector3::UnitZ());
  const HitPayload& mdt = factory.eta(StationName::BML, 0.);
  const HitPayload& bo = factory.eta(StationName::BOL, 0.);

  PatternState pat{candidate(seed, 0u), expSector(1u), &cfg, logger.get()};
  pat.addHit(candidate(rpcPhi, 1u), 2., 4.);
  BOOST_CHECK_EQUAL(static_cast<int>(pat.nPrecisionLayers), 1);
  BOOST_CHECK_EQUAL(static_cast<int>(pat.nTriggerLayers), 1);
  BOOST_CHECK_EQUAL(static_cast<int>(pat.nPhiLayers), 1);
  CHECK_SMALL(pat.patPhi, 1e-12);
  CHECK_CLOSE_REL(pat.patPhiCov, hitPhiCov, 1e-9);
  CHECK_CLOSE_REL(pat.meanNormResidual2, 0.25, 1e-12);
  CHECK_CLOSE_REL(pat.lastResidual, 2., 1e-12);
  CHECK_CLOSE_REL(pat.lastResSigma, 4., 1e-12);
  BOOST_CHECK(pat.lastInsertedHit == rpcPhi);
  BOOST_CHECK(pat.prevLayerHit == seed);
  BOOST_CHECK(pat.lineAnchorHit == seed);
  BOOST_CHECK(pat.needLineUpdate);
  BOOST_CHECK(pat.isInPattern(rpcPhi));

  /// Precision & trigger hits cannot replace each other
  BOOST_CHECK_THROW(pat.overWriteHit(candidate(mdt, 1u), 1., 2.),
                    std::runtime_error);
  /// Only the hit on the last layer can be replaced
  BOOST_CHECK_THROW(pat.overWriteHit(candidate(rpcEta, 2u), 1., 1.),
                    std::runtime_error);
  BOOST_CHECK_THROW(pat.overWriteHit(candidate(bo, 1u), 1., 1.),
                    std::runtime_error);
  BOOST_CHECK_EQUAL(static_cast<int>(pat.nPhiLayers), 1);

  pat.overWriteHit(candidate(rpcEta, 1u), 1., 1.);
  BOOST_CHECK_EQUAL(static_cast<int>(pat.nPrecisionLayers), 1);
  BOOST_CHECK_EQUAL(static_cast<int>(pat.nTriggerLayers), 1);
  BOOST_CHECK_EQUAL(static_cast<int>(pat.nPhiLayers), 0);
  /// The replaced residual is removed from the sum
  CHECK_CLOSE_REL(pat.meanNormResidual2, 1., 1e-12);
  const auto& bmHits = pat.hitsPerStation[stationIdx(MuonStationIndex::BM)];
  BOOST_CHECK_EQUAL(bmHits.size(), 1u);
  BOOST_CHECK(bmHits.back() == rpcEta);
  BOOST_CHECK(pat.lastInsertedHit == rpcEta);
  BOOST_CHECK(pat.prevLayerHit == seed);
  BOOST_CHECK(!pat.isInPattern(rpcPhi));
  /// Without phi hits, the pattern phi falls back to the sector
  CHECK_CLOSE_REL(pat.patPhiCov,
                  Acts::square(MuonSectorMapping::sectorSize(1)) / 3., 1e-12);

  /// sTgc pads can be replaced by strips on the same layer, but not vice versa
  const HitPayload& pad =
      factory.add(TechField::sTgc, StationName::BML, true, true, atPhi(0.),
                  tangent(0.), Acts::Vector3::UnitZ());
  const HitPayload& strip =
      factory.add(TechField::sTgc, StationName::BML, true, false, atPhi(0.),
                  tangent(0.), Acts::Vector3::UnitZ());
  PatternState padPat{candidate(seed, 0u), expSector(1u), &cfg, logger.get()};
  padPat.addHit(candidate(pad, 1u), 1., 1.);
  padPat.overWriteHit(candidate(strip, 1u), 1., 1.);
  BOOST_CHECK_EQUAL(static_cast<int>(padPat.nPrecisionLayers), 2);
  BOOST_CHECK_EQUAL(static_cast<int>(padPat.nTriggerLayers), 0);
  BOOST_CHECK_EQUAL(static_cast<int>(padPat.nPhiLayers), 0);
  BOOST_CHECK_THROW(padPat.overWriteHit(candidate(pad, 1u), 1., 1.),
                    std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
