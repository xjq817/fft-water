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
  // TODO: Loop until the user closes the window
  while (!glfwWindowShouldClose(window)) {
    glClearColor(97 / 256.f, 175 / 256.f, 239 / 256.f, 1.0f);

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
