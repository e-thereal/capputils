/*
 * tests.cu
 *
 *  Created on: Nov 4, 2011
 *      Author: tombr
 */

#include "tests.h"

#include <tbblas/device_vector.hpp>
#include <tbblas/device_matrix.hpp>

#include <boost/lambda/lambda.hpp>

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/timer.hpp>

namespace ublas = boost::numeric::ublas;

using namespace std;

int runtests() {
  using namespace boost::lambda;
  boost::timer timer;

//  cout << "tbblas" << endl;
//
//  float values[] = {
//      1, 2, 3, 4,
//      5, 6, 7, 8,
//      9, 10, 11, 12,
//      13, 14, 15, 16,
//      17, 18, 19, 20
//  };
//
//  float values1[] = {
//      1, 2,
//      3, 4
//  };
//
//  float values2[] = {
//      5, 6,
//      7, 8,
//      9, 10
//  };
//
//  ublas::matrix<float, ublas::row_major> m(5, 4), m1(2,2), m2(3,2);
//  copy(values, values + 20, m.data().begin());
//  copy(values1, values1 + 4, m1.data().begin());
//  copy(values2, values2 + 6, m2.data().begin());
//
//  cout << ublas::prod(ublas::trans(ublas::subrange(3.f * ublas::trans(m), 1,4, 2,4)), m2) / 2.f << endl;
//  cout << ublas::sum(ublas::column(m, 2)) << endl;
//
//  tbblas::device_matrix<float> dm(5, 4), dm1(2, 2), dm2(3, 2), dm3(3, 2);
//  dm = m;
//  dm1 = m1;
//  dm2 = m2;
//
//  (2.f * tbblas::subrange(dm3, 1,3, 0,2)) = tbblas::prod(tbblas::trans(tbblas::subrange(3.f * tbblas::trans(dm), 1,4, 2,4)), dm2);
//
//  cout << dm3.ublas() << endl;
//  cout << tbblas::sum(tbblas::column(dm, 2)) << endl;
//
//  ublas::column(m, 1) = ublas::column(m, 2) - 0.5f * ublas::column(m, 0);
//  cout << m << endl;
//  tbblas::column(dm, 1) = tbblas::column(dm, 2) - 0.5f * tbblas::column(dm, 0);
//  cout << dm.ublas() << endl;

  timer.restart();
  const size_t n = 1000;
  ublas::matrix<float, ublas::column_major> big1(n, n), big2(n, n), big3(n, n), big4(n, n);
  for_each(big1.data().begin(), big1.data().end(), _1 = (float)rand() / (float)RAND_MAX);
  for_each(big2.data().begin(), big2.data().end(), _1 = (float)rand() / (float)RAND_MAX);
  cout << "Init: " << timer.elapsed() << endl;
  timer.restart();
  //big3 = ublas::prod(big1, big2);
  cout << "uBLAS: " << timer.elapsed() << endl;
  timer.restart();
  tbblas::device_matrix<float> dbig1(n, n), dbig2(n, n), dbig3(n, n);
  dbig1 = big1;
  dbig2 = big2;
  cout << "Copy: " << timer.elapsed() << endl;
  timer.restart();
  for (int i = 0; i < 100; ++i)
    dbig3 = tbblas::prod(dbig1, dbig2);
  cudaThreadSynchronize();
  cout << "tbBLAS: " << timer.elapsed() / 100 << endl;

  big4 = dbig3;
  cout << "Error: " << ublas::norm_1(big3 - big4) << endl;

  return 0;
}
