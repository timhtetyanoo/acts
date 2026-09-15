// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Utilities/Logger.hpp"
#include "ActsExamples/EventData/MuonGlobalPatternFinderEvent.hpp"
#include "ActsExamples/EventData/MuonSpacePoint.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "ActsExamples/Framework/ProcessCode.hpp"
#include "ActsExamples/TrackFinding/MuonGlobalPatternFinder.hpp"

#include <memory>
#include <string>

namespace Acts {
class TrackingGeometry;
}  // namespace Acts

namespace ActsExamples {

/// @brief Algorithm performing the global pattern recognition in the muon
///        spectrometer, ported from the Athena MuonGlobalPatternFindingAlg.
///        It builds global patterns of precision and non-precision hits from
///        the space points of the event, first in eta and then adding the
///        compatible phi-only hits. The patterns are written to the event
///        store.
class MuonGlobalPatternFinding final : public IAlgorithm {
 public:
  /// @brief Configuration of the algorithm. It extends the configuration of
  ///        the pattern finder.
  struct Config : public MuonGlobalPatternFinder::Config {
    /// @brief Name of the input space point container
    std::string inputSpacePoints{};
    /// @brief Name of the output global pattern container
    std::string outputPatterns{};
    /// @brief Tracking geometry to transform the space points into the global
    ///        frame. Not needed if localToGlobal is set.
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry{};
    /// @brief Seed the patterns from the Inner station layer as well
    bool seedFromInner{false};
  };

  /// @brief Constructor
  /// @param cfg: Configuration of the algorithm
  /// @param logger: Logger
  explicit MuonGlobalPatternFinding(
      const Config& cfg, std::unique_ptr<const Acts::Logger> logger = nullptr);
  ~MuonGlobalPatternFinding() override;

  /// @brief Run the pattern finding on one event
  /// @param ctx: Algorithm context with the event information
  ProcessCode execute(const AlgorithmContext& ctx) const final;

  /// @brief Return the configuration
  const Config& config() const { return m_cfg; }
  /// @brief Return the pattern finder
  const MuonGlobalPatternFinder& patternFinder() const { return *m_finder; }

 private:
  Config m_cfg;
  std::unique_ptr<const MuonGlobalPatternFinder> m_finder{};

  ReadDataHandle<MuonSpacePointContainer> m_inputSpacePoints{
      this, "InputSpacePoints"};
  WriteDataHandle<MuonGlobalPatternContainer> m_outputPatterns{
      this, "OutputPatterns"};
};

}  // namespace ActsExamples
