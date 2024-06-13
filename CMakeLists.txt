cmake_minimum_required(VERSION 3.20)

project(OpenglStudy)
set(CMAKE_CXX_STANDARD 11)

# 查找 OpenGL 包
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

# 查找 OpenMP 包
find_package(OpenMP REQUIRED)

# 包含头文件目录
include_directories(
    /usr/include/glfw/3.4/include
    /usr/include/glew/2.2.0_1/include
    /usr/include/freeimage/3.18.0/include
    /usr/include/glm/1.0.1/include
    /usr/include/assimp/5.4.1/include
    /usr/include
    ./header
)

# 链接库文件目录
link_directories(
    /usr/lib
    /usr/local/lib
)

# 添加可执行文件
add_executable(main
    src/main.cpp
    src/fft.cpp
    src/timer.cpp
    src/common.cpp
    src/ocean.cpp
    src/skybox.cpp
    src/screenQuad.cpp
)

# 链接库文件
target_link_libraries(main
    ${OPENGL_gl_LIBRARY}
    GLEW
    glfw
    freeimage
    assimp
    OpenMP::OpenMP_CXX
    glut
)

# 在链接选项中添加 OpenGL 库
target_link_libraries(main ${OPENGL_LIBRARIES} GL GLU)

# 添加自定义命令（如果需要）
add_custom_target(cleanImg COMMAND rm -v ./result/*)
add_custom_target(cleanObj COMMAND rm -f *.o)

# 视频生成命令
add_custom_target(video
    COMMAND ffmpeg -r 30 -start_number 0 -i ../result/output/%04d.bmp -vcodec mpeg4 -b:v 30M -s 400x300 ../result/video.mp4
)
