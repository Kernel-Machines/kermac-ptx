cmake_minimum_required(VERSION 3.22)
project(kermac-ptx)

# Define the CUDA source file
set(CUDA_SOURCE_FILES src/test.cu)

# Create the ptx output directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/ptx)

# Custom command to generate PTX
foreach(CUDA_FILE ${CUDA_SOURCE_FILES})
    get_filename_component(CUDA_FILE_NAME ${CUDA_FILE} NAME_WE)
    set(PTX_OUTPUT "${CMAKE_BINARY_DIR}/ptx/${CUDA_FILE_NAME}.ptx")
    
    add_custom_command(
        OUTPUT ${PTX_OUTPUT}
        COMMAND nvcc -ptx 
            ${CMAKE_SOURCE_DIR}/${CUDA_FILE} 
            -o ${PTX_OUTPUT} 
            -arch=compute_89 
            -I${CMAKE_SOURCE_DIR}/thirdparty/cutlass/include
        DEPENDS ${CUDA_FILE}
        COMMENT "Generating PTX for ${CUDA_FILE}"
    )
    
    list(APPEND PTX_FILES ${PTX_OUTPUT})
endforeach()

# Custom target to trigger PTX generation
add_custom_target(generate_ptx ALL DEPENDS ${PTX_FILES})
