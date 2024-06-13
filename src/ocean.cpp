#include "ocean.h"
#include "common.h"

const float cOcean::BASELINE = 0.f;
vec2 cOcean::dudvMove = vec2(0.f, 0.f);

cOcean::cOcean(const int N, const float A, const vec2 w, const float length)
    : g(9.81), N(N), Nplus1(N + 1), A(A), w(w), length(length), vertices(0),
      h_tilde(0), h_tilde_slopex(0), h_tilde_slopez(0), h_tilde_dx(0),
      h_tilde_dz(0), fft(0) {
  // TODO: construct function

}

cOcean::~cOcean() {
  // TODO: destruct function
  
}

void cOcean::initShader() {
  shader = buildShader("../shader/vsOcean.glsl", "../shader/fsOcean.glsl",
                       "../shader/tcsQuad.glsl", "../shader/tesQuad.glsl");
}

void cOcean::initBuffers() {
  // for each mesh
  for (size_t i = 0; i < scene->mNumMeshes; i++) {
    const aiMesh *mesh = scene->mMeshes[i];
    int numVtxs = mesh->mNumVertices;

    // numVertices * numComponents
    GLfloat *aVtxCoords = new GLfloat[numVtxs * 3];
    GLfloat *aUvs = new GLfloat[numVtxs * 2];
    GLfloat *aNormals = new GLfloat[numVtxs * 3];

    for (size_t j = 0; j < numVtxs; j++) {
      aiVector3D &vtx = mesh->mVertices[j];
      aVtxCoords[j * 3 + 0] = vtx.x;
      aVtxCoords[j * 3 + 1] = vtx.y;
      aVtxCoords[j * 3 + 2] = vtx.z;

      aiVector3D &nml = mesh->mNormals[j];
      aNormals[j * 3 + 0] = nml.x;
      aNormals[j * 3 + 1] = nml.y;
      aNormals[j * 3 + 2] = nml.z;

      aiVector3D &uv = mesh->mTextureCoords[0][j];
      aUvs[j * 2 + 0] = uv.x;
      aUvs[j * 2 + 1] = uv.y;
    }

    // vao
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    vaos.push_back(vao);

    // vbo for vertex
    GLuint vboVtx;
    glGenBuffers(1, &vboVtx);
    glBindBuffer(GL_ARRAY_BUFFER, vboVtx);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * numVtxs * 3, aVtxCoords,
                 GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    vboVtxs.push_back(vboVtx);

    // vbo for uv
    GLuint vboUv;
    glGenBuffers(1, &vboUv);
    glBindBuffer(GL_ARRAY_BUFFER, vboUv);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * numVtxs * 2, aUvs,
                 GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    vboUvs.push_back(vboUv);

    // vbo for normal
    GLuint vboNml;
    glGenBuffers(1, &vboNml);
    glBindBuffer(GL_ARRAY_BUFFER, vboNml);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * numVtxs * 3, aNormals,
                 GL_STATIC_DRAW);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(2);
    vboNmls.push_back(vboNml);

    // delete client data
    delete[] aVtxCoords;
    delete[] aUvs;
    delete[] aNormals;
  } // end for each mesh
}

void cOcean::initTexture() {
  setTexture(tboDisp, 11, "../image/disp.png", FIF_PNG);
  setTexture(tboNormal, 12, "../image/normal.png", FIF_PNG);
  setTexture(tboFresnel, 13, "../image/fresnel.png", FIF_PNG);
  setTexture(tboPerlin, 14, "../image/perlin.png", FIF_PNG);
  setTexture(tboPerlinN, 15, "../image/perlinNormal.png", FIF_PNG);
  setTexture(tboPerlinDudv, 16, "../image/perlinDudv.png", FIF_PNG);
}

void cOcean::initUniform() {
  glUseProgram(shader);

  // transform
  uniM = myGetUniformLocation(shader, "M");
  uniV = myGetUniformLocation(shader, "V");
  uniP = myGetUniformLocation(shader, "P");

  // texture
  uniTexReflect = myGetUniformLocation(shader, "texReflect");
  uniTexRefract = myGetUniformLocation(shader, "texRefract");
  uniTexDisp = myGetUniformLocation(shader, "texDisp");
  uniTexNormal = myGetUniformLocation(shader, "texNormal");
  uniTexSkybox = myGetUniformLocation(shader, "texSkybox");
  uniTexFresnel = myGetUniformLocation(shader, "texFresnel");
  uniTexPerlin = myGetUniformLocation(shader, "texPerlin");
  uniTexPerlinN = myGetUniformLocation(shader, "texPerlinN");
  uniTexPerlinDudv = myGetUniformLocation(shader, "texPerlinDudv");

  glUniform1i(uniTexDisp, 11);
  glUniform1i(uniTexNormal, 12);
  glUniform1i(uniTexFresnel, 13);
  glUniform1i(uniTexPerlin, 14);
  glUniform1i(uniTexPerlinN, 15);
  glUniform1i(uniTexPerlinDudv, 16);
  glUniform1i(uniTexReflect, 3);
  glUniform1i(uniTexRefract, 2);

  // light
  uniLightColor = myGetUniformLocation(shader, "lightColor");
  uniLightPos = myGetUniformLocation(shader, "lightPos");

  // other
  uniDudvMove = myGetUniformLocation(shader, "dudvMove");
  uniEyePoint = myGetUniformLocation(shader, "eyePoint");
}

void cOcean::setTexture(GLuint &tbo, int texUnit, const string texDir,
                        FREE_IMAGE_FORMAT imgType) {
  glActiveTexture(GL_TEXTURE0 + texUnit);

  FIBITMAP *texImage =
      FreeImage_ConvertTo24Bits(FreeImage_Load(imgType, texDir.c_str()));

  glGenTextures(1, &tbo);
  glBindTexture(GL_TEXTURE_2D, tbo);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, FreeImage_GetWidth(texImage),
               FreeImage_GetHeight(texImage), 0, GL_BGR, GL_UNSIGNED_BYTE,
               (void *)FreeImage_GetBits(texImage));
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

  // release
  FreeImage_Unload(texImage);
}

float cOcean::dispersion(int n_prime, int m_prime) {
  // TODO
}

float cOcean::phillips(int n_prime, int m_prime) {
  // TODO
}

Complex cOcean::hTilde_0(int n_prime, int m_prime) {
  // TODO
}

Complex cOcean::hTilde(float t, int n_prime, int m_prime) {
  // TODO
}

void cOcean::evaluateWavesFFT(float t) {
  // TODO
}

void cOcean::render(float t, mat4 M, mat4 V, mat4 P, vec3 eyePoint,
                    vec3 lightColor, vec3 lightPos, bool resume, int frameN) {
  // TODO
}

vec3 cOcean::getVertex(int ix, int iz) {
  // TODO
}

void cOcean::writeHeightMap(int fNum) {
  int w, h;
  w = N;
  h = w;

  FIBITMAP *dispMap = FreeImage_Allocate(w, h, 24);
  RGBQUAD colorY;

  if (!dispMap) {
    std::cout << "FreeImage: Cannot allocate dispMap." << '\n';
    exit(EXIT_FAILURE);
  }

  // TODO

  FreeImage_Save(FIF_PNG, dispMap, "../image/disp.png", 0);
}

void cOcean::writeNormalMap(int fNum) {
  int w, h;
  w = N;
  h = w;

  FIBITMAP *bitmap = FreeImage_Allocate(w, h, 24);
  RGBQUAD color;

  if (!bitmap) {
    std::cout << "FreeImage: Cannot allocate image." << '\n';
    exit(EXIT_FAILURE);
  }

  // TODO

  FreeImage_Save(FIF_PNG, bitmap, "../image/normal.png", 0);
}

void cOcean::writeFoldingMap(int fNum) {
  int w, h;
  w = N;
  h = w;

  FIBITMAP *bitmap = FreeImage_Allocate(w, h, 24);
  RGBQUAD color;

  if (!bitmap) {
    std::cout << "FreeImage: Cannot allocate image." << '\n';
    exit(EXIT_FAILURE);
  }

  // TODO

  FreeImage_Save(FIF_PNG, bitmap, "../image/fold.png", 0);
}

float cOcean::Heaviside(float x) { return (x < 0.f) ? 0.f : 1.f; }

const char *cOcean::getFileDir(string prefix, int fNum) {
  // zero padding
  // e.g. "output0001.bmp"
  string num = to_string(fNum);
  num = string(4 - num.length(), '0') + num;
  string fileDir = prefix + num + ".png";

  return fileDir.c_str();
}

void cOcean::initReflect() {
  // framebuffer object
  glGenFramebuffers(1, &fboReflect);
  glBindFramebuffer(GL_FRAMEBUFFER, fboReflect);

  glActiveTexture(GL_TEXTURE0 + 3);
  glGenTextures(1, &tboReflect);
  glBindTexture(GL_TEXTURE_2D, tboReflect);

  // On OSX, must use WINDOW_WIDTH * 2 and WINDOW_HEIGHT * 2, don't know why
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WINDOW_WIDTH * 2, WINDOW_HEIGHT * 2, 0,
               GL_RGB, GL_UNSIGNED_BYTE, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, tboReflect, 0);

  // The depth buffer
  // User-defined framebuffer must have a depth buffer to enable depth test
  glGenRenderbuffers(1, &rboDepthReflect);
  glBindRenderbuffer(GL_RENDERBUFFER, rboDepthReflect);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, WINDOW_WIDTH * 2,
                        WINDOW_HEIGHT * 2);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                            GL_RENDERBUFFER, rboDepthReflect);

  glDrawBuffer(GL_COLOR_ATTACHMENT2);
}

void cOcean::initRefract() {
  // framebuffer object
  glGenFramebuffers(1, &fboRefract);
  glBindFramebuffer(GL_FRAMEBUFFER, fboRefract);

  glActiveTexture(GL_TEXTURE0 + 2);
  glGenTextures(1, &tboRefract);
  glBindTexture(GL_TEXTURE_2D, tboRefract);

  // On OSX, must use WINDOW_WIDTH * 2 and WINDOW_HEIGHT * 2, don't know why
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WINDOW_WIDTH * 2, WINDOW_HEIGHT * 2, 0,
               GL_RGB, GL_UNSIGNED_BYTE, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, tboRefract, 0);

  // The depth buffer
  // User-defined framebuffer must have a depth buffer to enable depth test
  glGenRenderbuffers(1, &rboDepthRefract);
  glBindRenderbuffer(GL_RENDERBUFFER, rboDepthRefract);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, WINDOW_WIDTH * 2,
                        WINDOW_HEIGHT * 2);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                            GL_RENDERBUFFER, rboDepthRefract);

  glDrawBuffer(GL_COLOR_ATTACHMENT1);
}

float uniformRandomVariable() {
  // TODO
}

Complex gaussianRandomVariable() {
  // TODO
}
