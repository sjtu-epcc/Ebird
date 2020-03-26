#benchmark
add_executable(batch_scheduler_test batch_scheduler_test.cc batch_scheduler_test.h)
target_link_libraries(batch_scheduler_test ebird ${THIRD_LIBS} -lpthread -lm)

add_executable(alexnet alexnet.cc alexnet.h)
target_link_libraries(alexnet ebird ${THIRD_LIBS} -lpthread -lm)

add_executable(inceptionv4 inceptionv4.cc inceptionv4.h)
target_link_libraries(inceptionv4 ebird ${THIRD_LIBS} -lpthread -lm)

add_executable(resnet_50 resnet_50.cc resnet_50.h)
target_link_libraries(resnet_50 ebird ${THIRD_LIBS} -lpthread -lm)

add_executable(resnet_101 resnet_101.cc resnet_101.h)
target_link_libraries(resnet_101 ebird ${THIRD_LIBS} -lpthread -lm)

add_executable(resnet_152 resnet_152.cc resnet_152.h)
target_link_libraries(resnet_152 ebird ${THIRD_LIBS} -lpthread -lm)

add_executable(vgg_16 vgg_16.cc vgg_16.h)
target_link_libraries(vgg_16 ebird ${THIRD_LIBS} -lpthread -lm)

add_executable(vgg_19 vgg_19.cc vgg_19.h)
target_link_libraries(vgg_19 ebird ${THIRD_LIBS} -lpthread -lm)

#profile
add_executable(profiler profiler.cc)
target_link_libraries(profiler ebird ${THIRD_LIBS} -lpthread -lm)

add_executable(prof_resnet_152 prof_resnet_152.cc resnet_152.h)
target_link_libraries(prof_resnet_152 ebird ${THIRD_LIBS} -lpthread -lm)

add_executable(one_resnet_152 one_resnet_152.cc resnet_152.h)
target_link_libraries(one_resnet_152 ebird ${THIRD_LIBS} -lpthread -lm)