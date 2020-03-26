
# link a shared library
CUDA_ADD_LIBRARY(ebird SHARED ${SOURCE_FILES} ${CUDA_FILES})
target_link_libraries(ebird ${THIRD_LIBS} -lcudnn -lpthread -lm)
