/*
 * fasttrainer.cu
 *
 *  Created on: Oct 29, 2013
 *      Author: tombr
 */

#include "tests.h"

#include <tbblas/tensor.hpp>
#include <tbblas/random.hpp>
#include <tbblas/math.hpp>
#include <tbblas/fft.hpp>
#include <tbblas/io.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/zeros.hpp>

#include <algorithm>

#include "mult_sum.hpp"
#include "repeat_mult.hpp"
#include "repeat_mult_sum.hpp"

using namespace tbblas;

typedef complex<float> complex_t;
typedef tbblas::tensor<float, 4, true> tensor_t;
typedef tbblas::tensor<complex<float>, 4, true> ctensor_t;
typedef tbblas::random_tensor<float, 4, true, normal<float> > randn_t;

typedef fft_plan<4> plan_t;

void fasttrainer(int size, int channelCount, int filterCount, int reps) {
  const int filterBatchSize = 4;
  assert(filterCount % filterBatchSize == 0);

  randn_t randv(size, size, size, channelCount);

  tensor_t v = floor(10 * randv) / 10;
  tensor_t F(size, size, size, channelCount * filterCount), f;
  float epsilonw = 0.001;

  for (int i = 0; i < (int)filterCount; ++i) {
    f = floor(10 * randv) / 10;
    F[seq(0,0,0,i * channelCount), f.size()] = f;
  }

  plan_t plan_v, plan_h, plan_f, plan_fb;

  ctensor_t cv = fft(v, 3, plan_v), cvneg, ch_full, ch,
      cF2 = fft(F, 3, plan_f),
      cFinc2 = zeros<complex_t>(cF2.size(), cF2.fullsize()), cH, ch2, cf, cvneg2;

  std::vector<ctensor_t> cF(filterCount / filterBatchSize), cFinc(filterCount / filterBatchSize);

  for (int i = 0; i < (int)cF.size(); ++i) {
    f = F[seq(0,0,0,i * channelCount * filterBatchSize), seq(size, size, size, channelCount * filterBatchSize)];
    cF[i] = fft(f, 3, plan_fb);
    cFinc[i] = zeros<complex_t>(cF[i].size(), cF[i].fullsize());
  }

  for (int iRep = 0; iRep < reps; ++iRep) {
    cH = conj_mult_sum(cv, cF2);
    cFinc2 += conj_repeat_mult(cv, cH, epsilonw);
    cvneg2 = repeat_mult_sum(cH, cF2);
  }

  for (int iRep = 0; iRep < reps; ++iRep) {
    cvneg = zeros<complex_t>(cv.size(), cv.fullsize());

    for (int iFilter = 0; iFilter < cF.size(); ++iFilter) {
//      ch_full = conj(cF[iFilter]) * cv;
//      ch = sum(ch_full, 3);
      ch = conj_mult_sum(cv, cF[iFilter]);

//      cFinc[iFilter] = cFinc[iFilter] + epsilonw * repeat(conj(ch), cv.size() / ch.size()) * cv;
      cFinc[iFilter] += conj_repeat_mult(cv, ch, epsilonw);
//      cvneg = cvneg + cF[iFilter] * repeat(ch, cF[iFilter].size() / ch.size());
      cvneg += repeat_mult_sum(ch, cF[iFilter]);

      if (reps == 1) {
        ch2 = cH[seq(0,0,0,iFilter * filterBatchSize), ch.size()];
        ch2.set_fullsize(ch.fullsize());
        tbblas_print(dot(ch - ch2, ch - ch2));

        cf = cFinc2[seq(0,0,0,iFilter * channelCount * filterBatchSize), cFinc[iFilter].size()];
        cf.set_fullsize(cFinc[iFilter].fullsize());
        tbblas_print(dot(cFinc[iFilter] - cf, cFinc[iFilter] - cf));

//        tbblas_print(cvneg[seq(0,0,2,4),seq(size/2+1,size,1,1)]);
      }
    }

    if (reps == 1) {
      tbblas_print(sqrt(abs(dot(cvneg - cvneg2, cvneg - cvneg2)) / cvneg.count()));
//      tbblas_print(cvneg2[seq(0,0,2,4),seq(size/2+1,size,1,1)]);
      tbblas_print(max(abs(cvneg - cvneg2)));
      tbblas_print(max((max(abs(cvneg - cvneg2)) == abs(cvneg - cvneg2)) * abs(cvneg)));
      tbblas_print(max((max(abs(cvneg - cvneg2)) == abs(cvneg - cvneg2)) * abs(cvneg2)));
    }
  }
}
