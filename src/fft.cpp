#include "fft.h"
#include "common.h"

cFFT::cFFT(unsigned int N) : N(N), re(0), W(0) {
  // finish TODO
  // logN
  logN = log(N) / log(2);

  // W
  W = new Complex* [logN];
  for (int i = 0, k = 1; i < logN; ++i, k <<= 1) {
    W[i] = new Complex [k];
    for (int j = 0; j < k; ++j)
      W[i][j] = Complex(cos(M_PI * j / k), sin(M_PI * j / k));
  }

  // reserved
  re = new unsigned int[N];
  for (int i = 0; i < N; ++i)
    re[i] = (re[i >> 1] >> 1) | ((i & 1) << (logN - 1));
  
  // C D
  C = new Complex[N];
  D = new Complex[N];
  which = 0;
}

cFFT::~cFFT() {
  // finish TODO
  // W
  if (W) {
    for (int i = 0; i < logN; ++i)
      if (W[i]) delete[] W[i];
    delete[] W;
  }

  // re
  if (re) delete[] re;
  
  // C
  if (C) delete[] C;
  if (D) delete[] D;
}

void cFFT::fft(Complex *A, Complex *B, int which, int offset) {
  // finish TODO
  // copy A into C
  for (int i = 0; i < N; ++i)
    C[i] = A[re[i] * which + offset];
  
  int loops = N >> 1;
  int size = 1;

  for (int i = 0; i < logN; ++i) {
#pragma omp parallel for num_threads(4)
    for (int j = 0; j < loops; ++j) {
      for (int k = 0; k < size; ++k) {
        D[(size << 1) * j + k] = C[(size << 1) * j + k] +
                                  C[(size << 1) * j + size + k] * W[i][k];
      }
      for (int k = size; k < (size << 1); ++k) {
        D[(size << 1) * j + k] = C[(size << 1) * j - size + k] -
                                  C[(size << 1) * j + k] * W[i][k - size];
      }
    }

    loops >>= 1;
    size <<= 1;

    for (int j = 0; j < N; ++j) C[j] = D[j];
  }

  for (int i = 0; i < N; ++i) B[i * which + offset] = C[i];
}
