add_subdirectory_if(Hashing ACTS_BUILD_EXAMPLES_HASHING)

add_unittest(MuonHoughSeeder MuonHoughSeederTests.cpp)
target_link_libraries(
    ActsUnitTestMuonHoughSeeder
    PRIVATE ActsExamplesTrackFinding ActsExamplesFramework
)

add_unittest(GlobalPatternFinder GlobalPatternFinderTests.cpp)
target_link_libraries(
    ActsUnitTestGlobalPatternFinder
    PRIVATE ActsExamplesTrackFinding ActsExamplesFramework
)

# Local end-to-end check on Athena-exported files, driven by the environment
# variables ACTS_GPF_NTUPLE and ACTS_GPF_GEOMETRY. Remove before upstreaming.
if(TARGET ActsExamplesIoRoot AND TARGET ActsPluginJson)
    add_unittest(GlobalPatternFinderData GlobalPatternFinderDataTests.cpp)
    target_link_libraries(
        ActsUnitTestGlobalPatternFinderData
        PRIVATE
            ActsExamplesTrackFinding
            ActsExamplesFramework
            ActsExamplesIoRoot
            ActsPluginJson
    )
endif()
