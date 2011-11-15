/*
 * tests.cu
 *
 *  Created on: Nov 4, 2011
 *      Author: tombr
 */
#include "tests.h"

#define BOOST_TYPEOF_COMPLIANT

#include <tbblas/device_vector.hpp>
#include <tbblas/device_matrix.hpp>
#include <boost/signals.hpp>
#include <boost/progress.hpp>
//#include <boost/any.hpp>

//#include <boost/lambda/lambda.hpp>

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/timer.hpp>

#include <thrust/transform.h>

namespace ublas = boost::numeric::ublas;

using namespace std;

// TODO: transform2d like thrust::transform but with 2d index
// TODO: sum(matrix) = vector containing colum sums
// TODO: sum(trans(matrix)) = vector containing row sums

int runtests() {
  //using namespace boost::lambda;
  boost::timer timer;

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

  tbblas::device_matrix<float> dm(5, 4), dm1(2, 2), dm2(3, 2), dm3(3, 2);
  dm = m;
  dm1 = m1;
  dm2 = m2;

  thrust::host_vector<float> v(2 * 3);
  tbblas::device_matrix<float> sdm = tbblas::subrange(dm, 1, 4, 1, 3);
  thrust::copy(sdm.begin(), sdm.end(), v.begin());
  for (int i = 0; i < v.size(); ++i)
    cout << v[i] << " ";
  cout << endl;

  tbblas::device_matrix<float> dm1000(1000,1000);
  timer.restart();
  for (int i = 0; i < 100; ++i)
    thrust::transform(dm1000.data().begin(), dm1000.data().end(), dm1000.data().begin(), dm1000.data().begin(), thrust::plus<float>());
  cout << timer.elapsed() << endl;

  timer.restart();
  for (int i = 0; i < 100; ++i)
    thrust::transform(dm1000.begin(), dm1000.end(), dm1000.begin(), dm1000.begin(), thrust::plus<float>());
  cout << timer.elapsed() << endl;

  return 0;
}
