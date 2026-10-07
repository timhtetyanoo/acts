set(
    unittest_extra_libraries
    ActsExamplesCudaGlobalPatternFinder
    ActsExamplesTrackFinding
)
add_unittest(CudaPatternFinderData CudaPatternFinderDataTests.cpp)
add_unittest(CudaPatternFinder CudaPatternFinderTests.cpp)
