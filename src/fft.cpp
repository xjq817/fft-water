#include "fft.h"
#include "common.h"

cFFT::cFFT(unsigned int N) : N(N), re(0), W(0), pi2(2 * M_PI) {
  // logN
  logN = log(N) / log(2);

  // reserved
  re = new unsigned int[N];
  for (int i = 0; i < N; ++i)
    re[i] = (re[i >> 1] >> 1) | ((i & 1) << (logN - 1));
  
}

cFFT::~cFFT() {
  // TODO
}

unsigned int cFFT::reverse(unsigned int i) {
  // TODO
}

Complex cFFT::w(unsigned int x, unsigned int N) {
  return Complex(cos(pi2 * x / N), sin(pi2 * x / N));
}

void cFFT::fft(Complex *input, Complex *output, int stride, int offset) {
  // TODO
}
