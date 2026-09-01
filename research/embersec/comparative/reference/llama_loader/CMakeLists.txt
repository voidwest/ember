cmake_minimum_required(VERSION 3.14)
project(embersec_loader_check LANGUAGES CXX)

if(NOT DEFINED LLAMA_CPP_SOURCE)
  message(FATAL_ERROR "Pass -DLLAMA_CPP_SOURCE=/path/to/llama.cpp")
endif()

# Build only the library needed by the loader harness.
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_CPP_SOURCE}" llama.cpp EXCLUDE_FROM_ALL)

add_executable(embersec_loader_check loader_check.cpp)
target_compile_features(embersec_loader_check PRIVATE cxx_std_17)
target_link_libraries(embersec_loader_check PRIVATE llama)
