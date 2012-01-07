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

//#include <boost/lambda/lambda.hpp>

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/timer.hpp>

#include <thrust/transform.h>
#include <thrust/reduce.h>

#include <fstream>
#include <sstream>

namespace ublas = boost::numeric::ublas;

using namespace std;

// TODO: transform2d like thrust::transform but with 2d index
// TODO: sum(trans(matrix)) = vector containing row sums

int runtests() {
  //using namespace boost::lambda;
  boost::timer timer;

  ifstream file("test.txt");
  if (file) {
    cout << "Reading file" << endl;

    string line;
    int rowCount = 0, columnCount = 0;
    std::vector<float> values;
    float value;
    while(getline(file, line)) {
      ++columnCount;
      stringstream lineStream(line);
      int rows = 0;
      while(!lineStream.eof()) {
        ++rows;
        lineStream >> value;
        values.push_back(value);
      }
      rowCount = std::max(columnCount, rows);
    }
    cout << "RowCount: " << rowCount << endl;
    cout << "ColumnCount: " << columnCount << endl;
    
    file.close();

    tbblas::device_matrix<float> dm(rowCount, columnCount);
    thrust::copy(values.begin(), values.end(), dm.data().begin());
  }

  return 0;

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

  dm1 *= tbblas::subrange(dm, 1, 3, 1, 3);
  cout << dm1.ublasColumnMajor() << endl;

  thrust::host_vector<float> v(2 * 3);
  tbblas::device_matrix<float> sdm = tbblas::subrange(dm, 1, 4, 1, 3);
  thrust::copy(sdm.begin(), sdm.end(), v.begin());
  for (int i = 0; i < v.size(); ++i)
    cout << v[i] << " ";
  cout << endl;

  tbblas::device_vector<float> csums(dm.size2());
  csums = tbblas::sum(dm);
  for (int i = 0; i < csums.size(); ++i)
    cout << csums(i) << " ";
  cout << endl;

#if 0
  cout << "n, prod, trans(all), trans(sub)" << endl;
  for (int n = 100; n < 1500; n *= 1.4) {
    const int reps = 2000 / n * 50;
    cout << n << ", ";
    tbblas::device_matrix<float> dm1000(n,n), dm4(n,n);
    timer.restart();
    for (int i = 0; i < reps; ++i)
      dm4 = tbblas::prod(dm1000, dm1000);
    cudaThreadSynchronize();
    cout << timer.elapsed() / reps << ", ";
    timer.restart();
    for (int i = 0; i < reps; ++i)
      thrust::transform(dm1000.data().begin(), dm1000.data().end(), dm1000.data().begin(), dm1000.data().begin(), thrust::plus<float>());
    cudaThreadSynchronize();
    cout << timer.elapsed() / reps << ", ";

    timer.restart();
    for (int i = 0; i < reps; ++i)
      thrust::transform(dm1000.begin(), dm1000.end(), dm1000.begin(), dm1000.begin(), thrust::plus<float>());
    cudaThreadSynchronize();
    cout << timer.elapsed() / reps << endl;
  }
  for (int n = 128; n <= 2048; n += 128) {
    const int reps = 2048 / n * 100;
      cout << n << ", ";
      tbblas::device_matrix<float> dm1000(n,n), dm4(n,n);
      timer.restart();
      for (int i = 0; i < reps; ++i)
        dm4 = tbblas::prod(dm1000, dm1000);
      cudaThreadSynchronize();
      cout << timer.elapsed() / reps << ", ";
      timer.restart();
      for (int i = 0; i < reps; ++i)
        thrust::transform(dm1000.data().begin(), dm1000.data().end(), dm1000.data().begin(), dm1000.data().begin(), thrust::plus<float>());
      cudaThreadSynchronize();
      cout << timer.elapsed() / reps << ", ";

      timer.restart();
      for (int i = 0; i < reps; ++i)
        thrust::transform(dm1000.begin(), dm1000.end(), dm1000.begin(), dm1000.begin(), thrust::plus<float>());
      cudaThreadSynchronize();
      cout << timer.elapsed() / reps << endl;
    }
#endif

  tbblas::device_matrix<double> doubleM(10, 10);

  return 0;
}
