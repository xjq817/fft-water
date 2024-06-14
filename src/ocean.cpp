#include "ocean.h"
#include "common.h"

const float cOcean::BASELINE = 0.f;
vec2 cOcean::dudvMove = vec2(0.f, 0.f);

cOcean::cOcean(const int N, const float A, const vec2 w, const float length)
    : g(9.81), N(N), Nplus1(N + 1), A(A), w(w), length(length), vertices(0),
      h_tilde(0), h_tilde_slopex(0), h_tilde_slopez(0), h_tilde_dx(0),
      h_tilde_dz(0), fft(0) {
  // finish TODO: construct function
  // malloc vertices
  vertices = new vertex_ocean[Nplus1 * Nplus1];

  for (int i = 0; i < Nplus1; ++i) {
    for (int j = 0; j < Nplus1; ++j) {
      int k = i * Nplus1 + j;

      // x, y, z and ox, oy, oz
      vertices[k].x = vertices[k].ox = length * (j - N / 2.0) / N;
      vertices[k].y = vertices[k].oy = 0.0;
      vertices[k].z = vertices[k].oz = length * (i - N / 2.0) / N;

      // nx, ny, nz
      vertices[k].x = 0.0;
      vertices[k].y = 1.0;
      vertices[k].z = 0.0;

      // a, b
      Complex h0 = hTilde_0(j, i);
      Complex hb = conj(hTilde_0(-j, -i));
      vertices[k].a = h0.real();
      vertices[k].b = h0.imag();
      vertices[k]._a = hb.real();
      vertices[k]._b = hb.imag();
    }
  }

  // malloc h_tilde
  h_tilde = new Complex[N * N];
  h_tilde_slopex = new Complex[N * N];
  h_tilde_slopez = new Complex[N * N];
  h_tilde_dx = new Complex[N * N];
  h_tilde_dz = new Complex[N * N];
  puts("???");

  // construct fft
  fft = new cFFT(N);

  // import water mesh
  puts("???");
  scene = importer.ReadFile("../mesh/gridQuad.obj", aiProcess_CalcTangentSpace);

  initShader();
  initBuffers();
  initTexture();
  initUniform();
  initReflect();
  initRefract();

  FIBITMAP *heightMap = FreeImage_Allocate(N, N, 24);
}

cOcean::~cOcean() {
  // finish TODO: destruct function
  delete[] vertices;
  delete[] h_tilde;
  delete[] h_tilde_slopex;
  delete[] h_tilde_slopez;
  delete[] h_tilde_dx;
  delete[] h_tilde_dz;
  delete fft;
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
  setTexture(tboFolding, 17, "../image/fold.png", FIF_PNG);
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
  uniTexFolding = myGetUniformLocation(shader, "texFolding");

  glUniform1i(uniTexDisp, 11);
  glUniform1i(uniTexNormal, 12);
  glUniform1i(uniTexFresnel, 13);
  glUniform1i(uniTexPerlin, 14);
  glUniform1i(uniTexPerlinN, 15);
  glUniform1i(uniTexPerlinDudv, 16);
  glUniform1i(uniTexFolding, 17);
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

  FIBITMAP *texImage = FreeImage_ConvertTo24Bits(FreeImage_Load(imgType, texDir.c_str()));

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
    //finish TODO
    // Define the base angular frequency
    float base_frequency = 2.0f * M_PI / 50.0f;
    
    // Calculate the wave number components in the x and z directions
    float kx = M_PI * (2 * n_prime - N) / length;
    float kz = M_PI * (2 * m_prime - N) / length;
    
    // Calculate the magnitude of the wave vector
    float k_magnitude = sqrt(kx * kx + kz * kz);
    
    // Calculate the angular frequency using the gravity constant
    float angular_frequency = sqrt(g * k_magnitude);
    
    // Normalize the frequency to the base frequency and round down
    float normalized_frequency = floor(angular_frequency / base_frequency) * base_frequency;
    
    return normalized_frequency;
}

float cOcean::phillips(int n_prime, int m_prime) {
  // finish TODO
  // calc L
  float wl = glm::length(w);
  float L = wl * wl / g;

  // calc k
  vec2 k = vec2(M_PI * (2 * n_prime - N) / length,
                M_PI * (2 * m_prime - N) / length);
  
  float kl = glm::length(k);
  if (kl < 1e-6) return 0.0;

  // calc dot(k, w)
  float kw = dot(normalize(k), normalize(w));
  float kw2 = kw * kw;

  // calc phillips
  float k2 = kl * kl, k4 = k2 * k2;
  float kL = kl * L, kL2 = kL * kL;

  return A * exp(-1.0 / kL2) / k4 * kw2 * exp(-kL2 * 1e-6);
}

Complex cOcean::hTilde_0(int n_prime, int m_prime) {
  //finish TODO
  Complex hOrigin = gaussianRandomVariable();
  // The wave amplitudes follow the desired energy distribution described by the Phillips spectrum.
  float phillipsSpectrum = phillips(n_prime, m_prime);
  return hOrigin * sqrt(phillipsSpectrum / 2.0);
}

Complex cOcean::hTilde(float t, int n_prime, int m_prime) {
  // finish TODO
  int k = m_prime * Nplus1 + n_prime;

  // calc omega
  float omega = dispersion(n_prime, m_prime);

  // calc exp
  float cos_ = cos(omega * t);
  float sin_ = sin(omega * t);

  Complex c0(cos_, sin_);
  Complex cb(cos_, -sin_);
  
  // calc h0 and conj_h0
  Complex h0 = Complex(vertices[k].a, vertices[k].b);
  Complex hb = Complex(vertices[k]._a, vertices[k]._b);

  return h0 * c0 + hb * cb;
}

void cOcean::evaluateWavesFFT(float t) {
  // finish TODO
  float kx, kz, len, lambda = -1.0f;
  int u, k;

#pragma omp parallel for private(kx, kz, len, u) collapse(2) num_threads(4)
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      // calc kz and kx
      kz = M_PI * (2.0 * i - N) / length;
      kx = M_PI * (2.0 * j - N) / length;
      len = sqrt(kx * kx + kz * kz);
      u = i * N + j;

      // calc h, slope and d
      h_tilde[u] = hTilde(t, j, i);
      h_tilde_slopex[u] = h_tilde[u] * Complex(0, kx);
      h_tilde_slopez[u] = h_tilde[u] * Complex(0, kz);
      if (len < 1e-6)
        h_tilde_dx[u] = h_tilde_dz[u] = Complex(0.0f, 0.0f);
      else {
        h_tilde_dx[u] = h_tilde[u] * Complex(0, -kx / len);
        h_tilde_dz[u] = h_tilde[u] * Complex(0, -kz / len);
      }
    }
  }

  // do fft for each row and column
  for (int i = 0; i < N; i++) {
    fft->fft(h_tilde, h_tilde, 1, i * N);
    fft->fft(h_tilde_slopex, h_tilde_slopex, 1, i * N);
    fft->fft(h_tilde_slopez, h_tilde_slopez, 1, i * N);
    fft->fft(h_tilde_dx, h_tilde_dx, 1, i * N);
    fft->fft(h_tilde_dz, h_tilde_dz, 1, i * N);
  }
  for (int j = 0; j < N; j++) {
    fft->fft(h_tilde, h_tilde, N, j);
    fft->fft(h_tilde_slopex, h_tilde_slopex, N, j);
    fft->fft(h_tilde_slopez, h_tilde_slopez, N, j);
    fft->fft(h_tilde_dx, h_tilde_dx, N, j);
    fft->fft(h_tilde_dz, h_tilde_dz, N, j);
  }

  double sign;
  vec3 n;

#pragma omp parallel for private(u, k, sign, n) collapse(2) num_threads(4)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      u = i * N + j;
      k = i * Nplus1 + j;
      sign = ((j + i) & 1) ? -1.0 : 1.0;

      // update x and z
      h_tilde_dx[u] = h_tilde_dx[u] * sign;
      h_tilde_dz[u] = h_tilde_dz[u] * sign;
      vertices[k].x = vertices[k].ox + h_tilde_dx[u].real() * lambda;
      vertices[k].z = vertices[k].oz + h_tilde_dz[u].real() * lambda;

      // update y
      h_tilde[u] = h_tilde[u] * sign;
      vertices[k].y = h_tilde[u].real();

      // update nx ny nz
      h_tilde_slopex[u] = h_tilde_slopex[u] * sign;
      h_tilde_slopez[u] = h_tilde_slopez[u] * sign;
      n = normalize(vec3(-h_tilde_slopex[u].real(), 1.0, -h_tilde_slopez[u].real()));
      vertices[k].nx = n.x;
      vertices[k].ny = n.y;
      vertices[k].nz = n.z;

      // update the edge
      if (j == 0 && i == 0) {
        vertices[k + N + Nplus1 * N].y = h_tilde[u].real();
        vertices[k + N + Nplus1 * N].x = vertices[k + N + Nplus1 * N].ox + h_tilde_dx[u].real() * lambda;
        vertices[k + N + Nplus1 * N].z = vertices[k + N + Nplus1 * N].oz + h_tilde_dz[u].real() * lambda;
        vertices[k + N + Nplus1 * N].nx = n.x;
        vertices[k + N + Nplus1 * N].ny = n.y;
        vertices[k + N + Nplus1 * N].nz = n.z;
      }
      if (j == 0) {
        vertices[k + N].y = h_tilde[u].real();
        vertices[k + N].x = vertices[k + N].ox + h_tilde_dx[u].real() * lambda;
        vertices[k + N].z = vertices[k + N].oz + h_tilde_dz[u].real() * lambda;
        vertices[k + N].nx = n.x;
        vertices[k + N].ny = n.y;
        vertices[k + N].nz = n.z;
      }
      if (i == 0) {
        vertices[k + Nplus1 * N].y = h_tilde[u].real();
        vertices[k + Nplus1 * N].x = vertices[k + Nplus1 * N].ox + h_tilde_dx[u].real() * lambda;
        vertices[k + Nplus1 * N].z = vertices[k + Nplus1 * N].oz + h_tilde_dz[u].real() * lambda;
        vertices[k + Nplus1 * N].nx = n.x;
        vertices[k + Nplus1 * N].ny = n.y;
        vertices[k + Nplus1 * N].nz = n.z;
      }
    }
  }
}

void cOcean::render(float t, mat4 M, mat4 V, mat4 P, vec3 eyePoint,
                    vec3 lightColor, vec3 lightPos, bool resume, int frameN) {
  //finish TODO
  if (resume) evaluateWavesFFT(t);

  writeHeightMap(frameN);
  writeNormalMap(frameN);
  writeFoldingMap(frameN);

  setTexture(tboDisp, 11, "../image/disp.png", FIF_PNG);
  setTexture(tboNormal, 12, "../image/normal.png", FIF_PNG);
  setTexture(tboNormal, 17, "../image/fold.png", FIF_PNG);

  // update transform matrix
  glUseProgram(shader);
  glUniformMatrix4fv(uniM, 1, GL_FALSE, value_ptr(M));
  glUniformMatrix4fv(uniV, 1, GL_FALSE, value_ptr(V));
  glUniformMatrix4fv(uniP, 1, GL_FALSE, value_ptr(P));

  glUniform3fv(uniEyePoint, 1, value_ptr(eyePoint));
  glUniform3fv(uniLightColor, 1, value_ptr(lightColor));
  glUniform3fv(uniLightPos, 1, value_ptr(lightPos));

  glUniform2fv(uniDudvMove, 1, value_ptr(dudvMove));

  // vec3(10.0, -0.1, 10.0) can produce ocean from a higher perspective
  mat4 Model = translate(mat4(1.0f), vec3(0, 0, 0));
  glUniformMatrix4fv(uniM, 1, GL_FALSE, value_ptr(Model));

  for (size_t i = 0; i < scene->mNumMeshes; i++) {
    int numVertex = scene->mMeshes[i]->mNumVertices;
    glBindVertexArray(vaos[i]);
    glDrawArrays(GL_PATCHES, 0, numVertex);
  }
}

vec3 cOcean::getVertex(int ix, int iz) {
  //finish TODO
  int index = iz * (N + 1) + ix;
  return vec3(vertices[index].x, vertices[index].y, vertices[index].z);
}

void cOcean::writeHeightMap(int fNum) {
  int w, h;
  w = h = N;

  FIBITMAP *dispMap = FreeImage_Allocate(w, h, 24);
  RGBQUAD colorDisp;

  // finish TODO
  for (int i = 0; i < w; i++){
    for (int j = 0; j < h; j++){
      int index = i * w + j;
      vertex_ocean vert = vertices[index];

      // scale x, y, z to [0, 1] consistent with shader
      float x = vert.ox - vert.x;
      x = (x + 2.f) / 10.f * 255.f; 
      float y = vert.oy - vert.y;
      y = (y + 2.f) / 10.f * 255.f;
      float z = vert.oz - vert.z;
      z = (z + 2.f) / 10.f * 255.f;

      colorDisp.rgbRed = int(x);
      colorDisp.rgbGreen = int(y);
      colorDisp.rgbBlue = int(z);
      FreeImage_SetPixelColor(dispMap, i, j, &colorDisp);
    }
  }

  FreeImage_Save(FIF_PNG, dispMap, "../image/disp.png", 0);
}

void cOcean::writeNormalMap(int fNum) {
  int w, h;
  w = h = N;

  FIBITMAP *bitmap = FreeImage_Allocate(w, h, 24);
  RGBQUAD color;

  // finish TODO
  for (int i = 0; i < w; i++){
    for (int j = 0; j < h; j++){
      int index = i * w + j;
      vertex_ocean ver = vertices[index];
      vec3 norrmal = normalize(vec3(ver.nx, ver.ny, ver.nz));
      norrmal = (norrmal + 1.f) / 2.f;  // [-1, -1] -> [0, 1]
      color.rgbRed = int(norrmal.x * 255.f);
      color.rgbGreen = int(norrmal.y * 255.f);
      color.rgbBlue = int(norrmal.z * 255.f);
      FreeImage_SetPixelColor(bitmap, i, j, &color);
    }
  }

  FreeImage_Save(FIF_PNG, bitmap, "../image/normal.png", 0);
}

void cOcean::writeFoldingMap(int fNum) {
  int w, h;
  w = h = N;

  FIBITMAP *bitmap = FreeImage_Allocate(w, h, 24);
  RGBQUAD color;

  // finish TODO
  for (int i = 0; i < w; i++){
    for (int j = 0; j < h; j++){
      int index = i * w + j;
      // four direction index
      int left =  (i + w - 1) % w;
      int right = (i + 1) % w;
      int up = (j + h - 1) % h;
      int down = (j + 1) % h;

      int indexLeft = left * w + j;
      int indexRight = right * w + j;
      int indexUp = i * w + up;
      int indexDown = i * w + down;

      float dxLeft = vertices[indexLeft].ox - vertices[indexLeft].x;
      float dxRight = vertices[indexRight].ox - vertices[indexRight].x;
      float dzUp = vertices[indexUp].oz - vertices[indexUp].z;
      float dzDown = vertices[indexDown].oz - vertices[indexDown].z;

      float Jxx = 1.f + (dxRight - dxLeft) / 2.f;
      float Jzz = 1.f + (dzUp - dzDown) / 2.f;
      float Jzx = (dxRight - dxLeft) / 2.f;
      float Jxz = Jzx;
      float Jaccobi = Jxx * Jzz - Jzx * Jxz;

      Jaccobi = glm::max(Jaccobi - 0.9f, 0.f);
      Jaccobi = Jaccobi * 255.f;
      Jaccobi = glm::min(Jaccobi, 255.f);
      color.rgbRed = color.rgbBlue = color.rgbGreen = int(Jaccobi);
      FreeImage_SetPixelColor(bitmap, i, j, &color);
    }
  }

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
  //finish TODO
  float f = 1.f * rand() / RAND_MAX;
  return f;
}

Complex gaussianRandomVariable() {
  //finish TODO
  float x1, x2, w=2.f;
  while (w >= 1.f){
    x1 = 2.f * uniformRandomVariable() - 1.f;
    x2 = 2.f * uniformRandomVariable() - 1.f;
    w = x1 * x1 + x2 * x2;
  }
  // Box-Muller transformation -> generate Gussian distribution random variable
  w = sqrt((-2.f * log(w)) / w);
  return Complex(x1 * w, x2 * w);
}
