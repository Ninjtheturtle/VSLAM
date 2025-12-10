cmake_minimum_required(VERSION 3.20)
project(VSLAM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# cmake tools sometimes passes -DCMAKE_CXX_COMPILER explicitly, which skips
# feature detection and leaves CMAKE_CXX_COMPILE_FEATURES empty on MSVC
if(MSVC AND NOT CMAKE_CXX_COMPILE_FEATURES)
  set(CMAKE_CXX_COMPILE_FEATURES
      "cxx_std_98;cxx_std_11;cxx_std_14;cxx_std_17;cxx_std_20;cxx_std_23"
      CACHE INTERNAL "" FORCE)
  set(CMAKE_CXX98_COMPILE_FEATURES  "cxx_std_98"  CACHE INTERNAL "" FORCE)
  set(CMAKE_CXX11_COMPILE_FEATURES  "cxx_std_11"  CACHE INTERNAL "" FORCE)
  set(CMAKE_CXX14_COMPILE_FEATURES  "cxx_std_14"  CACHE INTERNAL "" FORCE)
  set(CMAKE_CXX17_COMPILE_FEATURES  "cxx_std_17"  CACHE INTERNAL "" FORCE)
  set(CMAKE_CXX20_COMPILE_FEATURES  "cxx_std_20"  CACHE INTERNAL "" FORCE)
  set(CMAKE_CXX23_COMPILE_FEATURES  "cxx_std_23"  CACHE INTERNAL "" FORCE)
endif()

# RTX 3050 laptop = ampere SM86
set(CMAKE_CUDA_ARCHITECTURES 86)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# vcpkg deps
find_package(OpenCV REQUIRED COMPONENTS core features2d calib3d highgui video)
find_package(Ceres  REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

# rerun SDK (0.22.1) via FetchContent
# patch: rerun passes CMAKE_TOOLCHAIN_FILE to arrow's ExternalProject, which
# makes vcpkg try to install arrow's full vcpkg.json (grpc, protobuf, etc).
# we strip that line and add -DVCPKG_MANIFEST_MODE=OFF instead.
include(FetchContent)
FetchContent_Declare(rerun_sdk
  URL https://github.com/rerun-io/rerun/releases/download/0.22.1/rerun_cpp_sdk.zip
  DOWNLOAD_EXTRACT_TIMESTAMP NEW
)
FetchContent_GetProperties(rerun_sdk)
if(NOT rerun_sdk_POPULATED)
  FetchContent_Populate(rerun_sdk)
endif()

set(_arrow_cmake "${rerun_sdk_SOURCE_DIR}/download_and_build_arrow.cmake")
if(EXISTS "${_arrow_cmake}")
  file(READ "${_arrow_cmake}" _content)
  if(_content MATCHES "DCMAKE_TOOLCHAIN_FILE=")
    string(REGEX REPLACE
      "-DCMAKE_TOOLCHAIN_FILE=[^\r\n]*"
      "-DVCPKG_MANIFEST_MODE=OFF\n        -DARROW_S3=OFF"
      _content "${_content}")
    file(WRITE "${_arrow_cmake}" "${_content}")
    message(STATUS "patched arrow sub-cmake: disabled vcpkg toolchain")
  endif()
  unset(_content)
endif()
unset(_arrow_cmake)

add_subdirectory("${rerun_sdk_SOURCE_DIR}" "${rerun_sdk_BINARY_DIR}")

# hamming matcher CUDA kernel
add_library(hamming_matcher STATIC cuda/hamming_matcher.cu)
target_include_directories(hamming_matcher PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(hamming_matcher PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)
target_compile_options(hamming_matcher PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

# main executable
set(VSLAM_SOURCES
  src/main.cpp
  src/camera.cpp
  src/frame.cpp
  src/map_point.cpp
  src/map.cpp
  src/tracker.cpp
  src/local_ba.cpp
  src/visualizer.cpp
)

add_executable(vslam ${VSLAM_SOURCES})
target_include_directories(vslam PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(vslam PRIVATE
  hamming_matcher
  ${OpenCV_LIBS}
  Ceres::ceres
  Eigen3::Eigen
  rerun_sdk
)

# MSVC flags — guarded so they don't get passed to nvcc
if(MSVC)
  target_compile_options(vslam PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /MP /EHsc /bigobj /wd4244 /wd4305>
  )
  set_target_properties(vslam PROPERTIES
    VS_DEBUGGER_ENVIRONMENT "PATH=${CUDA_TOOLKIT_ROOT_DIR}/bin;$ENV{PATH}"
  )
endif()

install(TARGETS vslam DESTINATION bin)

message(STATUS "VSLAM: ${CMAKE_BUILD_TYPE} | SM${CMAKE_CUDA_ARCHITECTURES} | OpenCV ${OpenCV_VERSION} | Ceres ${CERES_VERSION}")
