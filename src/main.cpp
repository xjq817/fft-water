#include "common.h"
#include "ocean.h"
#include "skybox.h"
#include "screenQuad.h"

using namespace glm;
using namespace std;

void initGL();
void initOther();
void initMatrix();
void releaseResource();

void computeMatricesFromInputs();

GLFWwindow *window;
Skybox *skybox;
cOcean *ocean;
ScreenQuad *screenQuad;

bool saveTrigger = true;
int frameNumber = 0;
bool resume = true;
bool saveMap = true;

float verticalAngle = -1.79557;
float horizontalAngle = 3.16513;
float initialFoV = 45.0f;
float windSpeed = 16.0f;
float nearPlane = 0.01f, farPlane = 2000.f;

vec3 eyePoint = vec3(-36.338406, 1.624817, 1.602868);
vec3 eyeDirection =
    vec3(sin(verticalAngle) * cos(horizontalAngle), cos(verticalAngle),
         sin(verticalAngle) * sin(horizontalAngle));
vec3 up = vec3(0.f, 1.f, 0.f);

vec3 direction;

mat4 model, view, projection;
bool isRising = false, isDiving = false;

// for reflection texture
float verticalAngleReflect;
float horizontalAngleReflect;
vec3 eyePointReflect;
mat4 reflectV;

vec3 lightPos = vec3(0.f, 5.f, 0.f);
vec3 lightColor = vec3(1.f, 1.f, 1.f);
float lightPower = 12.f;

int N = 512;
float t = 0.f;

void reset_buffer(){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // clipping
    glEnable(GL_CLIP_DISTANCE0);
    glDisable(GL_CLIP_DISTANCE1);
}

int main(int argc, char *argv[]) {
  initGL();
  initOther();
  initMatrix();
  skybox = new Skybox();
  screenQuad = new ScreenQuad();
  // ocean simulator
  ocean = new cOcean(N, 0.005f, vec2(windSpeed, 0.0f), 16);

  // a rough way to solve cursor position initialization problem
  // must call glfwPollEvents once to activate glfwSetCursorPos
  // this is a glfw mechanism problem
  glfwPollEvents();
  glfwSetCursorPos(window, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);

  /* Loop until the user closes the window */
  // finish TODO: Loop until the user closes the window
  while (!glfwWindowShouldClose(window)) {
    glClearColor(97 / 256.f, 175 / 256.f, 239 / 256.f, 1.0f);

    computeMatricesFromInputs();

    glBindFramebuffer(GL_FRAMEBUFFER, ocean->fboRefract);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_CLIP_DISTANCE0);
    glDisable(GL_CLIP_DISTANCE1);

    vec4 clipplane_0 = vec4(0.f, -1.f, 0.f, sOcean::BASELINE);

    // draw skybox
    glEnable(GL_CULL_FACE);
    skybox->draw(model, view, projection, eyePoint);
    
    glBindFramebuffer(GL_FRAMEBUFFER, ocean->fboReflect);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // clipping
    glDisable(GL_CLIP_DISTANCE0);
    glEnable(GL_CLIP_DISTANCE1);

    vec4 clipplane_1 = vec4(0.f, 1.f, 0.f, cOcean::BASELINE + 0.125f);

    // draw skybox
    skybox->draw(model, reflectV, projection, eyePointReflect);

    /* render to main screen */
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glDisable(GL_CLIP_DISTANCE0);
    glDisable(GL_CLIP_DISTANCE1);
    // sky
    skybox->draw(model, view, projection, eyePoint);
    // ocean
    glDisable(GL_CULL_FACE);

    vec3 lightpos = eyePoint + vec3(direction.x * 4.0, 2.0, direction.z * 4.0);

    ocean->render(t, model, view, projection, eyePoint, lightColor, lightpos, resume, frameNumber);
    
    if (resume) {
      t += 0.01f;
    }
    // refresh frame
    glfwSwapBuffers(window);
    /* Poll for and process events */
    glfwPollEvents();

    cOcean::dudvMove += vec2(0.001, 0.001);
    if (saveTrigger) {
      string dir = "../result/output/";
      string num = to_string(frameNumber);
      num = string(4 - num.length(), '0') + num;
      string output = dir + num + ".bmp";

      FIBITMAP *outputImage = FreeImage_AllocateT(FIT_UINT32, WINDOW_WIDTH * 2, WINDOW_HEIGHT * 2);
      glReadPixels(0, 0, WINDOW_WIDTH * 2, WINDOW_HEIGHT * 2, GL_BGRA,GL_UNSIGNED_INT_8_8_8_8_REV, (GLvoid *)FreeImage_GetBits(outputImage));
      FreeImage_Save(FIF_BMP, outputImage, output.c_str(), 0);
      std::cout << output << " saved." << '\n';
    }

    frameNumber++;
  }

  // release
  releaseResource();

  return EXIT_SUCCESS;
}

void initGL() {
  // Initialise GLFW
  if (!glfwInit()) {
    fprintf(stderr, "Failed to initialize GLFW\n");
    getchar();
    exit(EXIT_FAILURE);
  }

  // without setting GLFW_CONTEXT_VERSION_MAJOR and _MINOR，
  // OpenGL 1.x will be used
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

  // must be used if OpenGL version >= 3.0
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Open a window and create its OpenGL context
  window =
      glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "FFT ocean", NULL, NULL);

  if (window == NULL) {
    std::cout << "Failed to open GLFW window." << std::endl;
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  glfwMakeContextCurrent(window);

  /* Initialize GLEW */
  // without this, glGenVertexArrays will report ERROR!
  glewExperimental = GL_TRUE;

  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    getchar();
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST); // must enable depth test!!
  glPatchParameteri(GL_PATCH_VERTICES, 4);
}


void initMatrix() {
  model = translate(mat4(1.f), vec3(0.f, 0.f, 0.f));
  view = lookAt(eyePoint, eyePoint + eyeDirection, up);
  projection = perspective(initialFoV, 1.f * WINDOW_WIDTH / WINDOW_HEIGHT,
                           nearPlane, farPlane);
}

void initOther() {
  // initialize random seed
  // this makes the ocean geometry different in every execution
  srand(clock());

  FreeImage_Initialise(true);
}

void releaseResource() {
  delete ocean;
  delete skybox;
  delete screenQuad;

  glfwTerminate();
  FreeImage_DeInitialise();
}
