cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(FHEViTInference)

set(CMAKE_CXX_STANDARD 17)
option(BUILD_STATIC "Set to ON to include static versions of the library" OFF)

# Find OpenFHE (make sure OpenFHE_DIR is set correctly)
find_package(OpenFHE CONFIG REQUIRED)

if(OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE include location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib location: ${OpenFHE_LIBDIR}")
else()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif()

# Flags from OpenFHE
set(CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")

# Include directories
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)

# Link directories
link_directories(${OpenFHE_LIBDIR})

if(BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries(${OpenFHE_STATIC_LIBRARIES})
else()
    set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    link_libraries(${OpenFHE_SHARED_LIBRARIES})
endif()

# 🔧 Main executable — change this if you're testing something else
#add_executable(fhe_inference
#    fhe_inference.cpp
 #   fhe_linear.cpp      # ⬅️ Add this
  #  weight_loader.cpp   # ⬅️ Also needed if not already added
#)
add_executable(plaintext
    plaintext.cpp
    fhe_linear.cpp      # ⬅️ Add this
    weight_loader.cpp   # ⬅️ Also needed if not already added
    transformer.cpp
)