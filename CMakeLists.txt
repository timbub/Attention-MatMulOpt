cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)

project(Attention-MatMul)

add_compile_options(
    $<$<CONFIG:Debug>:-fsanitize=address>
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
)

add_link_options(
    $<$<CONFIG:Debug>:-fsanitize=address>
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
)

add_executable(att src/main.cpp)

add_library(matrix_lib INTERFACE)
target_include_directories(matrix_lib INTERFACE ../Matrix_cpp/matrix)

add_library(att_lib STATIC
    src/Attention.cpp
    src/MatMuls.cpp
)

target_include_directories(att_lib PUBLIC inc)
target_link_libraries(att_lib PUBLIC matrix_lib)

target_link_libraries(att PRIVATE att_lib)
target_link_libraries(att PRIVATE matrix_lib)

target_compile_options(att_lib PRIVATE -O3 -mavx2 -mfma -march=native)
target_compile_options(att PRIVATE -O3 -mavx2 -mfma -march=native)





