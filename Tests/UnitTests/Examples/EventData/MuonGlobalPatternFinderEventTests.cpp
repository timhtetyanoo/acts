// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <boost/test/unit_test.hpp>

#include "ActsExamples/EventData/MuonGlobalPatternFinderEvent.hpp"
#include "ActsTests/CommonHelpers/FloatComparisons.hpp"

#include <numbers>
#include <stdexcept>
#include <vector>

using namespace ActsExamples;
using SectorProjector = MuonExpandedSector::SectorProjector;

namespace ActsExamples::Test {

namespace {

constexpr int nSectors = static_cast<int>(MuonSectorMapping::s_numSectors);
constexpr int nExpanded = 2 * nSectors;

constexpr unsigned nextSector(unsigned s) {
  return s == static_cast<unsigned>(nSectors) ? 1u : s + 1;
}

constexpr unsigned prevSector(unsigned s) {
  return s == 1u ? static_cast<unsigned>(nSectors) : s - 1;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(MuonGlobalPatternFinderEventTests)

BOOST_AUTO_TEST_CASE(SectorMapping) {
  for (int s = 1; s <= nSectors; ++s) {
    const double expectedSize =
        ((s % 2) != 0 ? 0.6 : 0.4) * MuonSectorMapping::s_eighthPi;
    CHECK_CLOSE_ABS(MuonSectorMapping::sectorSize(s), expectedSize, 1e-12);
    CHECK_CLOSE_ABS(
        MuonSectorMapping::sectorWidth(s),
        MuonSectorMapping::sectorSize(s) + MuonSectorMapping::s_sectorOverlap,
        1e-12);

    double expectedPhi = (s - 1) * MuonSectorMapping::s_eighthPi;
    if (expectedPhi > std::numbers::pi) {
      expectedPhi -= 2. * std::numbers::pi;
    }
    const double phi = MuonSectorMapping::sectorPhi(s);
    CHECK_CLOSE_ABS(phi, expectedPhi, 1e-12);
    BOOST_CHECK_EQUAL(MuonSectorMapping::getSector(phi), s);

    CHECK_CLOSE_ABS(MuonSectorMapping::transformPhiToSector(phi, s), 0., 1e-12);
    CHECK_CLOSE_ABS(
        MuonSectorMapping::transformPhiToSector(
            MuonSectorMapping::transformPhiToSector(phi, s, true), s, false),
        phi, 1e-12);

    BOOST_CHECK(MuonSectorMapping::insideSector(s, phi));
    const double outside = MuonSectorMapping::transformPhiToSector(
        MuonSectorMapping::sectorWidth(s) + 0.1, s, false);
    BOOST_CHECK(!MuonSectorMapping::insideSector(s, outside));

    std::vector<int> atCentre{};
    MuonSectorMapping::getSectors(phi, atCentre);
    BOOST_CHECK_EQUAL(atCentre.size(), 1u);
    BOOST_CHECK_EQUAL(atCentre.front(), s);

    CHECK_CLOSE_ABS(MuonSectorMapping::sectorOverlapPhi(s, s), phi, 1e-12);

    const int next = static_cast<int>(nextSector(static_cast<unsigned>(s)));
    const double overlap = MuonSectorMapping::sectorOverlapPhi(s, next);
    std::vector<int> atOverlap{};
    MuonSectorMapping::getSectors(overlap, atOverlap);
    BOOST_CHECK_EQUAL(atOverlap.size(), 2u);
  }

  CHECK_CLOSE_ABS(MuonSectorMapping::sectorOverlapPhi(1, 5), 0., 1e-12);
}

BOOST_AUTO_TEST_CASE(ExpandedSectorEncodeDecode) {
  for (unsigned s = 1; s <= static_cast<unsigned>(nSectors); ++s) {
    for (auto proj : {SectorProjector::center, SectorProjector::rightOverlap}) {
      const MuonExpandedSector es{s, proj};
      BOOST_CHECK_EQUAL(es.msSector(), s);
      BOOST_CHECK_EQUAL(es.projector(), proj);
    }

    const MuonExpandedSector left{s, SectorProjector::leftOverlap};
    const MuonExpandedSector right{prevSector(s),
                                   SectorProjector::rightOverlap};
    BOOST_CHECK_EQUAL(left, right);
    BOOST_CHECK_EQUAL(left.adjacentMsSector(), s);
  }

  for (int raw = 0; raw < nExpanded; ++raw) {
    const MuonExpandedSector es{static_cast<std::int8_t>(raw)};
    BOOST_CHECK(es.msSector() >= 1);
    BOOST_CHECK(es.msSector() <= static_cast<unsigned>(nSectors));
    const MuonExpandedSector roundTrip{es.msSector(), es.projector()};
    BOOST_CHECK_EQUAL(static_cast<int>(roundTrip.sector()), raw);
  }

  const MuonExpandedSector raw0{static_cast<std::int8_t>(0)};
  const MuonExpandedSector sec16Center{16u, SectorProjector::center};
  BOOST_CHECK_EQUAL(raw0, sec16Center);

  const MuonExpandedSector raw1{static_cast<std::int8_t>(1)};
  const MuonExpandedSector sec16Right{16u, SectorProjector::rightOverlap};
  BOOST_CHECK_EQUAL(raw1, sec16Right);
}

BOOST_AUTO_TEST_CASE(ExpandedSectorFromPhi) {
  for (unsigned s = 1; s <= static_cast<unsigned>(nSectors); ++s) {
    const MuonExpandedSector atCentre{MuonSectorMapping::sectorPhi(s)};
    BOOST_CHECK_EQUAL(atCentre.msSector(), s);
    BOOST_CHECK_EQUAL(atCentre.projector(), SectorProjector::center);

    const unsigned next = nextSector(s);
    const MuonExpandedSector atOverlap{
        MuonSectorMapping::sectorOverlapPhi(s, next)};
    BOOST_CHECK_EQUAL(atOverlap.msSector(), s);
    BOOST_CHECK_EQUAL(atOverlap.projector(), SectorProjector::rightOverlap);
    BOOST_CHECK_EQUAL(atOverlap.adjacentMsSector(), next);
  }
}

BOOST_AUTO_TEST_CASE(ExpandedSectorIsNeighbour) {
  const MuonExpandedSector raw0{static_cast<std::int8_t>(0)};
  const MuonExpandedSector raw1{static_cast<std::int8_t>(1)};
  const MuonExpandedSector raw2{static_cast<std::int8_t>(2)};
  const MuonExpandedSector raw31{static_cast<std::int8_t>(31)};

  BOOST_CHECK(raw0.isNeighbour(raw1));
  BOOST_CHECK(!raw0.isNeighbour(raw2));
  BOOST_CHECK(!raw0.isNeighbour(raw31));  // does not wrap around the ring
}

BOOST_AUTO_TEST_CASE(InvalidInputThrows) {
  BOOST_CHECK_THROW(MuonExpandedSector(std::int8_t{-1}), std::invalid_argument);
  BOOST_CHECK_THROW(MuonExpandedSector(std::int8_t{-5}), std::invalid_argument);
  BOOST_CHECK_THROW(MuonExpandedSector(std::int8_t{32}), std::invalid_argument);
  BOOST_CHECK_THROW(MuonExpandedSector(std::int8_t{127}),
                    std::invalid_argument);

  BOOST_CHECK_THROW(MuonExpandedSector(0u, SectorProjector::center),
                    std::invalid_argument);
  BOOST_CHECK_THROW(MuonExpandedSector(17u, SectorProjector::center),
                    std::invalid_argument);
  BOOST_CHECK_NO_THROW(MuonExpandedSector(1u, SectorProjector::center));
  BOOST_CHECK_NO_THROW(MuonExpandedSector(16u, SectorProjector::center));
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsExamples::Test
