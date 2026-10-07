set(
    unittest_extra_libraries
    ActsExamplesCudaGlobalPatternFinder
    ActsExamplesTrackFinding
)
add_unittest(CudaPatternFinderData CudaPatternFinderDataTests.cpp)
add_unittest(CudaPatternFinder CudaPatternFinderTests.cpp)

# Local end-to-end run on Athena-exported files, the GPU twin of
# ActsUnitTestGlobalPatternFinderData. Remove before upstreaming.
if(TARGET ActsExamplesIoRoot AND TARGET ActsPluginJson)
    add_unittest(CudaGlobalPatternFinderData CudaGlobalPatternFinderDataTests.cpp)
    target_link_libraries(
        ActsUnitTestCudaGlobalPatternFinderData
        PRIVATE ActsExamplesIoRoot ActsPluginJson
    )
endif()
