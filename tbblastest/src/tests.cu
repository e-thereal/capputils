/*
 * tests.cu
 *
 *  Created on: Nov 4, 2011
 *      Author: tombr
 */
#include "tests.h"

#define BOOST_TYPEOF_COMPLIANT

//#include <tbblas/device_vector.hpp>
//#include <tbblas/device_matrix.hpp>
#include <tbblas/device_tensor.hpp>
#include <tbblas/host_tensor.hpp>

#include <boost/signals.hpp>
#include <boost/progress.hpp>

//#include <boost/lambda/lambda.hpp>

//#include <boost/numeric/ublas/io.hpp>
//#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/timer.hpp>

#include <thrust/transform.h>
#include <thrust/reduce.h>

#include <fstream>
#include <sstream>

#include <curand.h>

//namespace ublas = boost::numeric::ublas;

using namespace std;

// TODO: transform2d like thrust::transform but with 2d index
// TODO: sum(trans(matrix)) = vector containing row sums

typedef tbblas::device_tensor<float, 3> tensor_t;
typedef tbblas::tensor_proxy<tensor_t::const_iterator, 3> const_proxy_t;

template<class T>
struct reference_test : thrust::binary_function<T, unsigned, T> {
  __device__
  T operator()(const T& value, const unsigned& i) const {
    const unsigned size = 3;
    T res = 0;
    for (unsigned k = 0; k < size; ++k)
      res += *(&value - (i % size) + k);
    return res;
  }
};

int runtests() {
  boost::timer timer;
  using namespace thrust::placeholders;

  tensor_t tensor(4, 3, 2), kernel(2, 2, 1), result(3, 2, 2), tensor2(5, 4, 2);

  float floats[] = {
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,

      10, 11, 12, 13,
      12, 14, 15, 16,
      16, 17, 18, 19
  };

  float fkernel[] = {
      1, 0,
      0, 0
  };
  thrust::copy(floats, floats + tensor.data().size(), tensor.data().begin());
  thrust::copy(fkernel, fkernel + kernel.data().size(), kernel.data().begin());


//  tensor = tbblas::flip(kernel) + 2.f * result;

  //timer.restart();
  //for (int i = 0; i < 1000; ++i)
    tbblas::flip(result) = tbblas::conv(kernel, tbblas::flip(tensor));
//    tensor = tbblas::conv(kernel, tensor2);
  //cout << "Time: " << timer.elapsed() << "ms" << endl;

//  thrust::device_vector<float> states(result.data().size());
//
//  curandGenerator_t gen;
//  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
//  curandGenerateUniform(gen, states.data().get(), states.size());
//  curandDestroyGenerator(gen);
//
//  thrust::transform(states.begin(), states.end(), thrust::make_counting_iterator(0), result.begin(), reference_test<float>());

  std::vector<float> vec(result.data().size());
  thrust::copy(result.begin(), result.end(), vec.begin());
  for (int i = 0; i < vec.size(); ++i)
    cout << vec[i] << " ";
  cout << endl;

#if 0
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

  return 0;
}
