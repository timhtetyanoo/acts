// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

/// @brief Local end-to-end check of the muon global pattern finder. It needs a
///        space point n-tuple and the matching tracking geometry, which are not
///        shipped with the repository, so both paths are taken from the
///        environment and the test is skipped when they are unset or missing:
///
///          ACTS_GPF_NTUPLE      space point n-tuple
///          ACTS_GPF_GEOMETRY    tracking geometry json
///          ACTS_GPF_OUTPUT      pattern output file, optional
///          ACTS_GPF_TIMING      per event timing csv, optional
///          ACTS_GPF_MAX_EVENTS  event cap, optional
///          ACTS_GPF_THREADS     thread counts of the throughput test,
///                               which runs only if it is set
///          ACTS_GPF_THROUGHPUT  throughput csv, optional
///
/// @note This file is a development aid and is not meant to be part of an
///       upstream pull request: a unit test does not read external data files.

#include <boost/test/unit_test.hpp>

#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/Framework/AlgorithmContext.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"
#include "ActsExamples/Io/Root/RootMuonGlobalPatternWriter.hpp"
#include "ActsExamples/Io/Root/RootMuonSpacePointReader.hpp"
#include "ActsExamples/TrackFinding/GlobalPatternFinderAlgorithm.hpp"
#include "ActsPlugins/Json/TrackingGeometryJsonConverter.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace ActsTests {

namespace {

/// @brief Returns the path stored in the environment variable or an empty path
///        if the variable is unset or points to a missing file
std::filesystem::path pathFromEnv(const char* variable) {
  const char* value = std::getenv(variable);
  if (value == nullptr) {
    BOOST_TEST_MESSAGE("Environment variable " << variable << " is not set.");
    return {};
  }
  std::filesystem::path path{value};
  if (!std::filesystem::exists(path)) {
    BOOST_TEST_MESSAGE("File " << path.string() << " from " << variable
                               << " does not exist.");
    return {};
  }
  return path;
}

/// @brief Returns the value of the environment variable or an empty string
std::string stringFromEnv(const char* variable) {
  const char* value = std::getenv(variable);
  return value != nullptr ? std::string{value} : std::string{};
}

/// @brief Returns the number stored in the environment variable or the fallback
///        if the variable is unset
std::size_t countFromEnv(const char* variable, std::size_t fallback) {
  const std::string value{stringFromEnv(variable)};
  if (value.empty()) {
    return fallback;
  }
  return static_cast<std::size_t>(std::stoul(value));
}

}  // namespace

BOOST_AUTO_TEST_SUITE(GlobalPatternFinderDataSuite)

BOOST_AUTO_TEST_CASE(pattern_finding_from_ntuple) {
  const std::filesystem::path ntuple{pathFromEnv("ACTS_GPF_NTUPLE")};
  const std::filesystem::path geometry{pathFromEnv("ACTS_GPF_GEOMETRY")};
  if (ntuple.empty() || geometry.empty()) {
    BOOST_TEST_MESSAGE("Skipping the data driven pattern finder test.");
    return;
  }

  const auto gctx = Acts::GeometryContext::dangerouslyDefaultConstruct();

  /// The space points are expressed in the frame of their spectrometer sector.
  /// Their measurement surfaces come from the tracking geometry
  Acts::TrackingGeometryJsonConverter geoConverter{};
  std::shared_ptr<Acts::TrackingGeometry> trackingGeometry{
      geoConverter.fromFile(gctx, geometry)};
  BOOST_REQUIRE(trackingGeometry != nullptr);

  ActsExamples::RootMuonSpacePointReader::Config readerCfg{};
  readerCfg.filePath = ntuple.string();
  readerCfg.treeName = "MuonSpacePoints";
  readerCfg.outputSpacePoints = "MuonSpacePoints";
  ActsExamples::RootMuonSpacePointReader reader{readerCfg, Acts::Logging::INFO};

  ActsExamples::GlobalPatternFinderAlgorithm::Config finderCfg{};
  finderCfg.inSpacePoints = readerCfg.outputSpacePoints;
  finderCfg.outPatterns = "MuonGlobalPatterns";
  finderCfg.trackingGeometry = trackingGeometry;
  /// Seeds from MDT hits and from the inner station, if set to non-zero
  finderCfg.seedFromMdt = countFromEnv("ACTS_GPF_SEED_FROM_MDT", 0u) != 0u;
  finderCfg.seedFromInner = countFromEnv("ACTS_GPF_SEED_FROM_INNER", 0u) != 0u;
  ActsExamples::GlobalPatternFinderAlgorithm finder{
      finderCfg, Acts::getDefaultLogger("GlobalPatternFinderAlgorithm",
                                        Acts::Logging::DEBUG)};

  ActsExamples::ReadDataHandle<ActsExamples::MuonGlobalPatternContainer>
      patternHandle{&finder, "TestOutputPatterns"};
  patternHandle.initialize(finderCfg.outPatterns);

  /// The wall time of the pattern finding, one row per event. The reader and
  ///  the writer are separate calls, so the measurement covers the algorithm
  ///  alone. `steady_clock` is monotonic, so a change of the system time leaves
  ///  the durations intact. The hit counts belong on the row because they tell
  ///  a busier event apart from a slower one.
  ActsExamples::ReadDataHandle<ActsExamples::MuonSpacePointContainer>
      spacePointHandle{&finder, "TestInputSpacePoints"};
  spacePointHandle.initialize(readerCfg.outputSpacePoints);
  std::ofstream timingFile{};
  if (const std::string timingPath{stringFromEnv("ACTS_GPF_TIMING")};
      !timingPath.empty()) {
    timingFile.open(timingPath);
    timingFile << "event,nSpacePoints,nBuckets,nPatterns,totalTime_us\n";
  }

  /// The patterns are written out for the offline validation, if asked for
  std::unique_ptr<ActsExamples::RootMuonGlobalPatternWriter> writer{};
  const std::string outFile{stringFromEnv("ACTS_GPF_OUTPUT")};
  if (!outFile.empty()) {
    ActsExamples::RootMuonGlobalPatternWriter::Config writerCfg{};
    writerCfg.inputPatterns = finderCfg.outPatterns;
    writerCfg.inputSpacePoints = readerCfg.outputSpacePoints;
    writerCfg.filePath = outFile;
    writer = std::make_unique<ActsExamples::RootMuonGlobalPatternWriter>(
        writerCfg, Acts::Logging::INFO);
  }

  const auto [firstEvent, availableEvents] = reader.availableEvents();
  BOOST_REQUIRE_GT(availableEvents, firstEvent);
  const std::size_t lastEvent{std::min(
      availableEvents,
      firstEvent +
          countFromEnv("ACTS_GPF_MAX_EVENTS", availableEvents - firstEvent))};
  BOOST_TEST_MESSAGE("Reading the events [" << firstEvent << ", " << lastEvent
                                            << ") from " << ntuple.string());

  std::size_t totalPatterns{0};
  std::size_t eventsWithPatterns{0};
  for (std::size_t event = firstEvent; event < lastEvent; ++event) {
    /// Each event starts from an empty store, as the sequencer would do
    ActsExamples::WhiteBoard eventStore{};
    ActsExamples::AlgorithmContext ctx{0, event, eventStore, 0};

    BOOST_REQUIRE(reader.read(ctx) == ActsExamples::ProcessCode::SUCCESS);

    const auto startedAt = std::chrono::steady_clock::now();
    BOOST_REQUIRE(finder.execute(ctx) == ActsExamples::ProcessCode::SUCCESS);
    const auto elapsed = std::chrono::steady_clock::now() - startedAt;

    const ActsExamples::MuonGlobalPatternContainer& patterns{
        patternHandle(ctx)};
    if (writer) {
      BOOST_REQUIRE(writer->write(ctx) == ActsExamples::ProcessCode::SUCCESS);
    }
    if (timingFile.is_open()) {
      const ActsExamples::MuonSpacePointContainer& spacePoints{
          spacePointHandle(ctx)};
      std::size_t nSpacePoints{0};
      for (const ActsExamples::MuonSpacePointBucket& bucket : spacePoints) {
        nSpacePoints += bucket.size();
      }
      timingFile << event << ',' << nSpacePoints << ',' << spacePoints.size()
                 << ',' << patterns.size() << ','
                 << std::chrono::duration_cast<std::chrono::microseconds>(
                        elapsed)
                        .count()
                 << '\n';
    }
    totalPatterns += patterns.size();
    eventsWithPatterns += (patterns.empty() ? 0u : 1u);

    BOOST_TEST_MESSAGE("Event " << event << ": found " << patterns.size()
                                << " global patterns.");
    for (const ActsExamples::MuonGlobalPattern& pattern : patterns) {
      std::size_t nHits{0};
      for (const auto& hits : pattern.hitsPerStation) {
        nHits += hits.size();
      }
      BOOST_TEST_MESSAGE(
          "  Pattern in sector "
          << static_cast<int>(pattern.sector) << ", theta: " << pattern.theta
          << ", phi: " << pattern.phi << ", hits: " << nHits
          << ", precision/trigger/phi layers: " << pattern.nPrecisionLayers
          << "/" << pattern.nTriggerLayers << "/" << pattern.nPhiLayers);
      /// Every pattern has to satisfy the cuts it was selected with
      BOOST_CHECK_GE(pattern.nPrecisionLayers, finderCfg.minPrecisionLayers);
      BOOST_CHECK_GT(nHits, 0u);
    }
  }

  if (writer) {
    BOOST_REQUIRE(writer->finalize() == ActsExamples::ProcessCode::SUCCESS);
    BOOST_TEST_MESSAGE("Wrote the patterns to " << outFile);
  }

  BOOST_TEST_MESSAGE("Found " << totalPatterns << " patterns in "
                              << eventsWithPatterns << " of "
                              << (lastEvent - firstEvent) << " events.");
  BOOST_CHECK_GT(totalPatterns, 0u);
}

/// @brief Events per second of the finder on several threads, run when
///        ACTS_GPF_THREADS lists thread counts, e.g. "1 2 4 8 12". The events
///        are read into memory first, so reading is not measured; each thread
///        then takes the next event until none is left. A pass on one thread
///        gives each event's pattern count, which every later pass has to
///        reproduce. Each thread count runs the events once untimed, to warm
///        up, then once timed. A row per thread count goes to the standard
///        output and to ACTS_GPF_THROUGHPUT, if given.
BOOST_AUTO_TEST_CASE(pattern_finding_throughput) {
  const std::string threadList{stringFromEnv("ACTS_GPF_THREADS")};
  const std::filesystem::path ntuple{pathFromEnv("ACTS_GPF_NTUPLE")};
  const std::filesystem::path geometry{pathFromEnv("ACTS_GPF_GEOMETRY")};
  if (threadList.empty() || ntuple.empty() || geometry.empty()) {
    BOOST_TEST_MESSAGE("Skipping the throughput test.");
    return;
  }

  const auto gctx = Acts::GeometryContext::dangerouslyDefaultConstruct();
  Acts::TrackingGeometryJsonConverter geoConverter{};
  std::shared_ptr<Acts::TrackingGeometry> trackingGeometry{
      geoConverter.fromFile(gctx, geometry)};
  BOOST_REQUIRE(trackingGeometry != nullptr);

  ActsExamples::RootMuonSpacePointReader::Config readerCfg{};
  readerCfg.filePath = ntuple.string();
  readerCfg.treeName = "MuonSpacePoints";
  readerCfg.outputSpacePoints = "MuonSpacePoints";
  ActsExamples::RootMuonSpacePointReader reader{readerCfg, Acts::Logging::INFO};

  ActsExamples::GlobalPatternFinderAlgorithm::Config finderCfg{};
  finderCfg.inSpacePoints = readerCfg.outputSpacePoints;
  finderCfg.outPatterns = "MuonGlobalPatterns";
  finderCfg.trackingGeometry = trackingGeometry;
  finderCfg.seedFromMdt = countFromEnv("ACTS_GPF_SEED_FROM_MDT", 0u) != 0u;
  finderCfg.seedFromInner = countFromEnv("ACTS_GPF_SEED_FROM_INNER", 0u) != 0u;
  ActsExamples::GlobalPatternFinderAlgorithm finder{
      finderCfg, Acts::getDefaultLogger("GlobalPatternFinderAlgorithm",
                                        Acts::Logging::WARNING)};
  ActsExamples::ReadDataHandle<ActsExamples::MuonGlobalPatternContainer>
      patternHandle{&finder, "TestOutputPatterns"};
  patternHandle.initialize(finderCfg.outPatterns);

  const auto [firstEvent, availableEvents] = reader.availableEvents();
  BOOST_REQUIRE_GT(availableEvents, firstEvent);
  const std::size_t nEvents{std::min(
      availableEvents - firstEvent,
      countFromEnv("ACTS_GPF_MAX_EVENTS", availableEvents - firstEvent))};
  std::vector<ActsExamples::WhiteBoard> inputs{};
  inputs.reserve(nEvents);
  for (std::size_t i = 0u; i < nEvents; ++i) {
    ActsExamples::AlgorithmContext ctx{0, firstEvent + i, inputs.emplace_back(),
                                       0};
    BOOST_REQUIRE(reader.read(ctx) == ActsExamples::ProcessCode::SUCCESS);
  }
  BOOST_REQUIRE(finder.initialize() == ActsExamples::ProcessCode::SUCCESS);

  struct Pass {
    double wallTime{0.};
    double cpuTime{0.};
    std::size_t failures{0u};
    std::size_t overflows{0u};
  };
  std::vector<std::size_t> nPatterns(nEvents, 0u);
  /// Runs every event once on nThreads threads and fills nPatterns
  auto runPass = [&](std::size_t nThreads) {
    std::atomic<std::size_t> next{0u};
    std::atomic<std::size_t> failures{0u};
    std::atomic<std::size_t> overflows{0u};
    auto work = [&](std::size_t thread) {
      for (std::size_t i = next++; i < nEvents; i = next++) {
        // A fresh store sharing the event's space points, as the sequencer
        // gives each event
        ActsExamples::WhiteBoard store{};
        store.copyFrom(inputs[i]);
        ActsExamples::AlgorithmContext ctx{0, firstEvent + i, store, thread};
        try {
          if (finder.execute(ctx) != ActsExamples::ProcessCode::SUCCESS) {
            ++failures;
            continue;
          }
          nPatterns[i] = patternHandle(ctx).size();
        } catch (const std::exception&) {
          ++failures;
        }
      }
    };
    const std::clock_t cpuStart{std::clock()};
    const auto start = std::chrono::steady_clock::now();
    {
      std::vector<std::jthread> threads{};
      for (std::size_t t = 0u; t < nThreads; ++t) {
        threads.emplace_back(work, t);
      }
    }
    const std::chrono::duration<double> wall{std::chrono::steady_clock::now() -
                                             start};
    return Pass{.wallTime = wall.count(),
                .cpuTime = static_cast<double>(std::clock() - cpuStart) /
                           CLOCKS_PER_SEC,
                .failures = failures.load(),
                .overflows = overflows.load()};
  };

  runPass(1u);
  const std::vector<std::size_t> reference{nPatterns};

  std::ofstream outFile{};
  if (const std::string outPath{stringFromEnv("ACTS_GPF_THROUGHPUT")};
      !outPath.empty()) {
    outFile.open(outPath);
  }
  const std::string header{
      "threads,events,wallTime_s,eventsPerSecond,cpuTime_s,coresUsed,"
      "overflows,failures,mismatches"};
  std::cout << header << std::endl;
  if (outFile.is_open()) {
    outFile << header << '\n';
  }
  std::istringstream threadCounts{threadList};
  for (std::size_t nThreads{0u}; threadCounts >> nThreads;) {
    BOOST_REQUIRE_GT(nThreads, 0u);
    runPass(nThreads);
    std::ranges::fill(nPatterns, 0u);
    const Pass pass{runPass(nThreads)};
    std::size_t mismatches{0u};
    for (std::size_t i = 0u; i < nEvents; ++i) {
      mismatches += (nPatterns[i] != reference[i]) ? 1u : 0u;
    }
    std::ostringstream row{};
    row << nThreads << ',' << nEvents << ',' << pass.wallTime << ','
        << static_cast<double>(nEvents) / pass.wallTime << ',' << pass.cpuTime
        << ',' << pass.cpuTime / pass.wallTime << ',' << pass.overflows << ','
        << pass.failures << ',' << mismatches;
    std::cout << row.str() << std::endl;
    if (outFile.is_open()) {
      outFile << row.str() << '\n';
    }
    BOOST_CHECK_EQUAL(pass.failures, 0u);
    BOOST_CHECK_EQUAL(mismatches, 0u);
  }
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
