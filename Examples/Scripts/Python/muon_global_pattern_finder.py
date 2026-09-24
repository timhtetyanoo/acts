#!/usr/bin/env python3
import argparse

import acts
import acts.examples
from acts.examples.root import RootMuonSpacePointReader
from acts.json import TrackingGeometryJsonConverter


def runGlobalPatternFinder(
    inFile: str, geoFile: str, nEvents: int, logLevel: acts.logging.Level
):
    # The muon space points are stored in the frame of their spectrometer sector.
    # Reaching the global frame needs the measurement surfaces, which are read
    # from the tracking geometry JSON written alongside the n-tuple.
    gctx = acts.GeometryContext.dangerouslyDefaultConstruct()
    trackingGeometry = TrackingGeometryJsonConverter().fromFile(gctx, geoFile)

    # The sequencer drives the event loop: for every event it calls each
    # reader, then each algorithm, in the order they were added. Data is
    # passed between them through the event store under string keys.
    s = acts.examples.Sequencer(events=nEvents, numThreads=1, logLevel=logLevel)

    # Reads the space points and writes them into the event store as a
    # MuonSpacePointContainer under the key given by outputSpacePoints. The tree
    # holding them is called "MuonSpacePoints", which differs from the reader's
    # default.
    spReader = RootMuonSpacePointReader(
        filePath=inFile,
        treeName="MuonSpacePoints",
        outputSpacePoints="MuonSpacePoints",
        level=logLevel,
    )
    s.addReader(spReader)

    # Reads the space points back via inSpacePoints and writes the found
    # patterns under outPatterns. Every Config field not set here keeps its
    # default from the C++ header.
    patternFinder = acts.examples.GlobalPatternFinderAlgorithm(
        inSpacePoints=spReader.config.outputSpacePoints,
        outPatterns="MuonGlobalPatterns",
        trackingGeometry=trackingGeometry,
        level=logLevel,
    )
    s.addAlgorithm(patternFinder)

    s.run()


if "__main__" == __name__:
    p = argparse.ArgumentParser(
        description="Run the muon global pattern finder on muon space points",
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to the ROOT n-tuple with the MuonSpacePoints tree",
    )
    p.add_argument(
        "--geometry",
        required=True,
        help="Path to the tracking geometry JSON matching the n-tuple",
    )
    p.add_argument("--nEvents", default=100, type=int, help="Number of events to run")
    p.add_argument(
        "--logLevel",
        default="INFO",
        choices=["VERBOSE", "DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level of the sequencer, reader and algorithm",
    )

    args = p.parse_args()
    runGlobalPatternFinder(
        inFile=args.input,
        geoFile=args.geometry,
        nEvents=args.nEvents,
        logLevel=getattr(acts.logging, args.logLevel),
    )
