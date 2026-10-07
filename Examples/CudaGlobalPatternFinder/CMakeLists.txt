acts_add_library(
    ExamplesCudaGlobalPatternFinder
    src/CudaPatternFinderData.cpp
    src/CudaPatternFinder.cu
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

# The kernels reproduce the CPU numbers bit for bit only without contraction
# into fused multiply-adds
target_compile_options(
    ActsExamplesCudaGlobalPatternFinder
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>
)

acts_compile_headers(ExamplesCudaGlobalPatternFinder GLOB "include/**/*.hpp")
