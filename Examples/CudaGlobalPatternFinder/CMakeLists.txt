acts_add_library(
    ExamplesCudaGlobalPatternFinder
    src/CudaPatternFinderData.cpp
    src/CudaPatternFinder.cu
    src/CudaPatternFinderV2.cu
    src/CudaPatternFinder.cpp
    src/CudaGlobalPatternFinderV2.cpp
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
        ActsExamplesTrackFinding
        CUDA::cudart
)

# The version 1 kernels reproduce the CPU numbers bit for bit only without
# contraction into fused multiply-adds. Version 2 is judged on the physics and
# keeps the compiler's default
set_source_files_properties(
    src/CudaPatternFinder.cu
    PROPERTIES COMPILE_OPTIONS -fmad=false
)

acts_compile_headers(ExamplesCudaGlobalPatternFinder GLOB "include/**/*.hpp")
