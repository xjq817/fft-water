#ifndef OCEAN_H
#define OCEAN_H

#include <complex>

#include "timer.h"
#include "fft.h"
#include "common.h"

typedef complex<double> Complex;

float uniformRandomVariable();
Complex gaussianRandomVariable();

struct vertex_ocean {
  GLfloat x, y, z;    // vertex
  GLfloat nx, ny, nz; // normal
  GLfloat a, b;    // htilde0
  GLfloat _a, _b; // htilde0mk conjugate
  GLfloat ox, oy, oz; // original position
};

class cOcean {
public:
  float g;          // gravity constant
  int N, Nplus1;    // dimension -- N should be a power of 2
  float A;          // phillips spectrum parameter -- affects heights of waves
  vec2 w;           // wind parameter
  float length;     // length parameter
  Complex *h_tilde, // for fast fourier transform
      *h_tilde_slopex, *h_tilde_slopez, *h_tilde_dx, *h_tilde_dz;
  cFFT *fft;              // fast fourier transform
  vertex_ocean *vertices; // vertices info for simulation

  /* vertices info for rendering */
  Assimp::Importer importer;
  const aiScene *scene;

  static const float BASELINE;
  static vec2 dudvMove;

  vector<GLuint> vboVtxs, vboUvs, vboNmls;
  vector<GLuint> vaos;

  GLuint shader;
  GLuint tboDisp, tboNormal, tboFresnel, tboFolding;
  GLuint tboPerlin, tboPerlinN, tboPerlinDudv;
  GLint uniM, uniV, uniP;
  GLint uniLightColor, uniLightPos;
  GLint uniTexReflect, uniTexRefract, uniTexDisp, uniTexNormal, uniTexSkybox, uniTexFolding;
  GLint uniTexPerlin, uniTexPerlinN, uniTexPerlinDudv, uniTexFresnel;
  GLint uniEyePoint;
  GLint uniDudvMove;
  GLuint tboRefract, tboReflect;
  GLuint fboRefract, fboReflect;
  GLuint rboDepthRefract, rboDepthReflect;

  FIBITMAP *dispMap, *normalMap;

protected:
public:
  cOcean(const int N, const float A, const vec2 w, const float length);
  ~cOcean();

  float dispersion(int n_prime, int m_prime); // deep water
  float phillips(int n_prime, int m_prime);   // phillips spectrum
  Complex hTilde_0(int n_prime, int m_prime);
  Complex hTilde(float t, int n_prime, int m_prime);
  void evaluateWavesFFT(float t);
  void render(float, mat4, mat4, mat4, vec3, vec3, vec3, bool, int);
  vec3 getVertex(int ix, int iz);

  void initBuffers();
  void initShader();
  void initTexture();
  void initUniform();
  void initReflect();
  void initRefract();
  void setTexture(GLuint &, int, const string, FREE_IMAGE_FORMAT);
  void writeHeightMap(int);
  void writeNormalMap(int);
  void writeFoldingMap(int);
  float Heaviside(float);
  const char *getFileDir(string, int);
};

#endif
