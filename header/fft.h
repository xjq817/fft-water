#ifndef FFT_H
#define FFT_H

#include <math.h>
#include <complex>

using namespace std;

typedef complex<double> Complex;

class cFFT {
private:
  unsigned int N, which;
  unsigned int logN;
  unsigned int *re;
  Complex **W;
  Complex *C, *D;

protected:
public:
  cFFT(unsigned int N);
  ~cFFT();
  void fft(Complex *input, Complex *output, int stride, int offset);
};

#endif
