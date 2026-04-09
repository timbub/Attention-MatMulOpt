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

add_executable(att main.cpp Attention.cpp MatMuls.cpp)

add_library(matrix_lib INTERFACE)
target_include_directories(matrix_lib INTERFACE ../Matrix_cpp/matrix)

target_link_libraries(att PRIVATE matrix_lib)




