acts_add_library(
    ExamplesCudaGlobalPatternFinder
    src/CudaPatternFinderData.cpp
    ACTS_INCLUDE_FOLDER include/ActsExamples
)

target_include_directories(
    ActsExamplesCudaGlobalPatternFinder
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(
    ActsExamplesCudaGlobalPatternFinder
    PUBLIC
        Acts::ExamplesFramework
        ActsExamplesCudaMuonSegmentFitting
        CUDA::cudart
)

acts_compile_headers(ExamplesCudaGlobalPatternFinder GLOB "include/**/*.hpp")
