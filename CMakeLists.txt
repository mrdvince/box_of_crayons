cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(plants VERSION 1.0.0 LANGUAGES CXX)

set(EXECUTABLE_NAME plants)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED PATHS ~/libtorch)
find_package(OpenCV REQUIRED PATHS)


add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE main.cpp
                                          src/model.cpp
                                          src/trainer.cpp
                                          src/dataset.cpp
                                          include/trainer.h
                                          include/model.h
                                          include/dataset.h
)

target_include_directories(${EXECUTABLE_NAME} PRIVATE include)

target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)


option(CONVERT_MODEL "Script model" ON)

if(CONVERT_MODEL)
    execute_process(COMMAND python ${CMAKE_CURRENT_LIST_DIR}/convert.py)
endif()

if(MSVC)
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR})
    include(copy_torch_dlls)
    copy_torch_dlls(${EXECUTABLE_NAME})
endif(MSVC)