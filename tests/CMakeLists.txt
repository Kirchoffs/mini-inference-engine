aux_source_directory(. TESTS_DIRS)
aux_source_directory(../src/data SRC_DIRS)
aux_source_directory(../src/layers SRC_DIRS)
aux_source_directory(../src/ops SRC_DIRS)

add_executable(test_mini_inference_engine ${TESTS_DIRS} ${SRC_DIRS})

target_include_directories(test_mini_inference_engine PRIVATE 
    ../include
    ${ARMADILLO_INCLUDE_DIRS}
)

target_link_libraries(test_mini_inference_engine PRIVATE 
    glog::glog
    GTest::gtest
    ${ARMADILLO_LIBRARIES}
    pthread
)
