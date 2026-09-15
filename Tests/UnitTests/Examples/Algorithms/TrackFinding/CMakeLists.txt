add_subdirectory_if(Hashing ACTS_BUILD_EXAMPLES_HASHING)

add_unittest(MuonHoughSeeder MuonHoughSeederTests.cpp)
target_link_libraries(
    ActsUnitTestMuonHoughSeeder
    PRIVATE ActsExamplesTrackFinding ActsExamplesFramework
)

add_unittest(MuonGlobalPatternFinderUtils MuonGlobalPatternFinderUtilsTests.cpp)
target_link_libraries(
    ActsUnitTestMuonGlobalPatternFinderUtils
    PRIVATE ActsExamplesTrackFinding ActsExamplesFramework
)

add_unittest(MuonGlobalPatternFinderDefs MuonGlobalPatternFinderDefsTests.cpp)
target_link_libraries(
    ActsUnitTestMuonGlobalPatternFinderDefs
    PRIVATE ActsExamplesTrackFinding ActsExamplesFramework
)
