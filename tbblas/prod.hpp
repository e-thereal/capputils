/*
 * prod.hpp
 *
 *  Created on: Nov 20, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_PROD_HPP_
#define TBBLAS_PROD_HPP_

#include <tbblas/tensor.hpp>
#include <tbblas/proxy.hpp>
#include <tbblas/type_traits.hpp>
#include <tbblas/assert.hpp>

namespace tbblas {

template<class T, class Tcp, class Tp>
void gemm(bool transa, bool transb, int m, int n, int k, T alpha, Tcp A, int lda,
    Tcp B, int ldb, T beta, Tp C, int ldc);

template<>
void gemm(bool transa, bool transb, int m, int n, int k, float alpha, thrust::device_ptr<const float> A, int lda,
    thrust::device_ptr<const float> B, int ldb, float beta, thrust::device_ptr<float> C, int ldc);

template<>
void gemm(bool transa, bool transb, int m, int n, int k, double alpha, thrust::device_ptr<const double> A, int lda,
    thrust::device_ptr<const double> B, int ldb, double beta, thrust::device_ptr<double> C, int ldc);

template<>
void gemm(bool transa, bool transb, int m, int n, int k, float alpha, const float* A, int lda,
    const float* B, int ldb, float beta, float* C, int ldc);

template<>
void gemm(bool transa, bool transb, int m, int n, int k, double alpha, const double* A, int lda,
    const double* B, int ldb, double beta, double* C, int ldc);

template<class Proxy>
struct prod_operation {
  typedef typename Proxy::value_t value_t;
  typedef typename Proxy::dim_t dim_t;

  static const unsigned dimCount = Proxy::dimCount;
  static const bool cuda_enabled = Proxy::cuda_enabled;

  typedef tensor<value_t, dimCount, cuda_enabled> tensor_t;

  prod_operation(const Proxy& proxy1, const Proxy& proxy2)
   : _proxy1(proxy1), _proxy2(proxy2)
  {
    tbblas_assert(proxy1.size() == proxy1.fullsize());
    tbblas_assert(proxy2.size() == proxy2.fullsize());

    _size[0] = proxy1.size()[0];
    _size[1] = proxy2.size()[1];
  }

  void apply(tensor_t& output) const {
    bool trans1 = _proxy1.order()[0] == 1;
    bool trans2 = _proxy2.order()[0] == 1;

    dim_t size1 = _proxy1.size(), size2 = _proxy2.size();
    dim_t start1 = _proxy1.start(), start2 = _proxy2.start();
    dim_t pitch1 = _proxy1.pitch(), pitch2 = _proxy2.pitch();

    gemm(trans1, trans2, size1[0], size2[1], size1[1], value_t(1),
        &_proxy1.data()[0] + start1[1] * pitch1[0] + start1[0], pitch1[0],
        &_proxy2.data()[0] + start2[1] * pitch2[0] + start2[0], pitch2[0],
        value_t(0), &output.data()[0], _size[0]);
  }

  void apply_inc(tensor_t& output) const {
    bool trans1 = _proxy1.order()[0] == 1;
    bool trans2 = _proxy2.order()[0] == 1;

    dim_t size1 = _proxy1.size(), size2 = _proxy2.size();
    dim_t start1 = _proxy1.start(), start2 = _proxy2.start();
    dim_t pitch1 = _proxy1.pitch(), pitch2 = _proxy2.pitch();

    gemm(trans1, trans2, size1[0], size2[1], size1[1], value_t(1),
        &_proxy1.data()[0] + start1[1] * pitch1[0] + start1[0], pitch1[0],
        &_proxy2.data()[0] + start2[1] * pitch2[0] + start2[0], pitch2[0],
        value_t(1), &output.data()[0], _size[0]);
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _size;
  }

private:
  Proxy _proxy1, _proxy2;
  dim_t _size;
};

template<class T>
struct is_operation<prod_operation<T> > {
  static const bool value = true;
};

template<class T>
struct is_inc_operation<prod_operation<T> > {
  static const bool value = true;
};

template<class Proxy>
typename boost::enable_if<is_proxy<Proxy>,
    typename boost::enable_if_c<Proxy::dimCount == 2,
      prod_operation<Proxy>
    >::type
>::type
prod(const Proxy& proxy1, const Proxy& proxy2) {
  tbblas_assert(proxy1.size()[1] == proxy2.size()[0]);
  return prod_operation<Proxy>(proxy1, proxy2);
}

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
    typename boost::enable_if_c<Tensor::dimCount == 2,
      prod_operation<proxy<Tensor> >
    >::type
>::type
prod(const proxy<Tensor>& proxy1, Tensor& tensor2) {
  return prod(proxy1, proxy<Tensor>(tensor2));
}

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
    typename boost::enable_if_c<Tensor::dimCount == 2,
      prod_operation<proxy<Tensor> >
    >::type
>::type
prod(Tensor& tensor1, const proxy<Tensor>& proxy2) {
  return prod(proxy<Tensor>(tensor1), proxy2);
}

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
    typename boost::enable_if_c<Tensor::dimCount == 2,
      prod_operation<proxy<Tensor> >
    >::type
>::type
prod(Tensor& tensor1, Tensor& tensor2) {
  return prod(proxy<Tensor>(tensor1), proxy<Tensor>(tensor2));
}

}

#endif /* TBBLAS_PROD_HPP_ */
