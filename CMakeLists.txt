cmake_minimum_required(VERSION 3.14)
project(portrait_mode)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(OpenCV REQUIRED)
find_package(nng REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/deps/nngpp/include ${EIGEN_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/deps/concurrentqueue)

SET(FAST_BIL_SOURCES
        src/bilateral_grid_simplified.cpp
        src/bilateral_grid_simplified.h
        src/fast_bilateral_solver.cpp
        src/fast_bilateral_solver.h
        src/stereo_matcher_birchfield_tomasi.cpp
        src/stereo_matcher_birchfield_tomasi.h
        )

SET(PORTRAIT_MODE_SOURCES
        src/main.cpp
        src/DepthMapActor.cpp
        src/DepthMapActor.h
        src/BlurActor.cpp
        src/BlurActor.h
        src/DepthMapImpl.h
        src/FacialRecognitionActor.cpp
        src/FacialRecognitionActor.h
        src/FeatureMatchingActor.cpp
        src/FeatureMatchingActor.h
        src/BlurImpl.h
        src/FastBILDepthMapImpl.cpp
        src/FastBILDepthMapImpl.h
        src/GaussianBlurImpl.cpp
        src/GaussianBlurImpl.h
        src/BoxBlurImpl.h
        src/BoxBlurImpl.cpp
        src/ParGaussianBlurImpl.cpp
        src/ParGaussianBlurImpl.h
        src/MonoDepth2Impl.cpp
        src/MonoDepth2Impl.h
        src/StereoSGBMImpl.cpp
        src/StereoSGBMImpl.h
        src/ParDiscBlurImpl.cpp
        src/ParDiscBlurImpl.h)

add_executable(portrait_mode ${PORTRAIT_MODE_SOURCES} ${FAST_BIL_SOURCES})
target_link_libraries(portrait_mode ${OpenCV_LIBS} nng ${CERES_LIBRARIES} ${GLOG_LIBRARIES})