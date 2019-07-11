# Target name
set(target Oreo)

# Define
set(root_dir ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Includes
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Libraries
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Includes
)

# Sources
file(GLOB header_dir
        ${root_dir}/Includes/c)

file(GLOB_RECURSE headers
        ${header_dir}/*.hpp
        ${header_dir}/Util/*.hpp)

file(GLOB_RECURSE includes ${root_dir}/Includes/*.hpp)

file(GLOB_RECURSE sources
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build library
add_library(${target}
        ${sources} ${includes})


# Project options
set_target_properties(${target}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        ${DEFAULT_PROJECT_OPTIONS}
        )

# Compile options
target_compile_options(${target}
        PRIVATE

        PUBLIC
        ${DEFAULT_COMPILE_OPTIONS}

        INTERFACE
        )

target_link_libraries(${target}
        PRIVATE

        PUBLIC
        ${DEFAULT_LINKER_OPTIONS}
        ${DEFAULT_LIBRARIES}

        INTERFACE
        )

# Install
install(TARGETS ${target} DESTINATION lib)
install(DIRECTORY ${header_dir} DESTINATION include)