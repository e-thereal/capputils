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

namespace ublas = boost::numeric::ublas;

using namespace std;

int runtests() {
  using namespace boost::lambda;

  cout << "tbblas" << endl;

  float values[] = {
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16,
      17, 18, 19, 20
  };

  float values1[] = {
      1, 2,
      3, 4
  };

  float values2[] = {
      5, 6,
      7, 8,
      9, 10
  };

  ublas::matrix<float, ublas::row_major> m(5, 4), m1(2,2), m2(3,2);
  copy(values, values + 20, m.data().begin());
  copy(values1, values1 + 4, m1.data().begin());
  copy(values2, values2 + 6, m2.data().begin());

  cout << ublas::prod(ublas::trans(ublas::subrange(3.f * ublas::trans(m), 1,4, 2,4)), m2) / 2.f << endl;
  cout << ublas::sum(ublas::column(m, 2)) << endl;

  tbblas::device_matrix<float> dm(5, 4), dm1(2, 2), dm2(3, 2), dm3(3, 2);
  dm = m;
  dm1 = m1;
  dm2 = m2;
  (2.f * tbblas::subrange(dm3, 1,3, 0,2)) = tbblas::prod(tbblas::trans(tbblas::subrange(3.f * tbblas::trans(dm), 1,4, 2,4)), dm2);

  cout << dm3.ublas() << endl;
  cout << tbblas::sum(tbblas::column(dm, 2)) << endl;

  ublas::column(m, 1) = ublas::column(m, 2) - 0.5f * ublas::column(m, 0);
  cout << m << endl;
  tbblas::column(dm, 1) = tbblas::column(dm, 2) - 0.5f * tbblas::column(dm, 0);
  cout << dm.ublas() << endl;

  return 0;
}
