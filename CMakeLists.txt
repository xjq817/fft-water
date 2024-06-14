cmake_minimum_required (VERSION 3.20)

project (OpenglStudy)
set(CMAKE_CXX_STANDARD 11)
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

# glfw

# brew 安装的库都在下面的这个目录里 /usr/local/Cellar

# 把glfw加载项目
include_directories(/opt/homebrew/Cellar/glfw/3.4/include
                    /opt/homebrew/Cellar/glew/2.2.0_1/include
                    /opt/homebrew/Cellar/freeimage/3.18.0/include
                    /opt/homebrew/Cellar/glm/1.0.1/include
                    /opt/homebrew/Cellar/assimp/5.4.1/include
                    /opt/homebrew/Cellar/libomp/18.1.6/include
                    /opt/homebrew/Cellar/freeglut/3.4.0/include
                    ./header

)

# 库文件也加载来 里面就是 glfw 的库文件了
link_directories(   /opt/homebrew/Cellar/glfw/3.4/lib
                    /opt/homebrew/Cellar/glew/2.2.0_1/lib
                    /opt/homebrew/Cellar/freeimage/3.18.0/lib
                    /opt/homebrew/Cellar/assimp/5.4.1/lib
                    /opt/homebrew/Cellar/libomp/18.1.6/lib
                    /opt/homebrew/Cellar/freeglut/3.4.0/lib
)

add_executable(main
    src/main.cpp
    src/fft.cpp
    src/timer.cpp
    src/common.cpp
    src/ocean.cpp
    src/skybox.cpp
    src/screenQuad.cpp
)
target_link_libraries(main ${OPENGL_gl_LIBRARY})
target_link_libraries(main "-framework Cocoa")


target_link_libraries(main
    GLEW
    glfw
    freeimage
    assimp
    omp
    GLUT
)

# 添加自定义命令（如果需要）
add_custom_target(cleanImg COMMAND rm -v ./result/*)
add_custom_target(cleanObj COMMAND rm -f *.o)

# 视频生成命令
add_custom_target(video
    COMMAND ffmpeg -r 30 -start_number 0 -i ../result/output/%04d.bmp -vcodec mpeg4 -b:v 30M -s 400x300 ../result/video.mp4
)
