/*
 * sum2.hpp
 *
 *  Created on: Oct 1, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_SUM_HPP_
#define TBBLAS_SUM_HPP_

#include <tbblas/type_traits.hpp>
#include <tbblas/proxy.hpp>

#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

#include <boost/utility/enable_if.hpp>

namespace tbblas {

/* Sum of all elements */

template<class Expression>
typename boost::enable_if<is_expression<Expression>,
  typename Expression::value_t
>::type
sum(const Expression& expr) {
  return thrust::reduce(expr.begin(), expr.end(), typename Expression::value_t());
}

/* Sum along one dimension */

template<class Proxy>
struct sum_operation {
  typedef typename Proxy::value_t value_t;
  typedef typename Proxy::dim_t dim_t;

  static const unsigned dimCount = Proxy::dimCount;

  typedef tensor<value_t, dimCount, Proxy::cuda_enabled> tensor_t;

  template <typename T1>
  struct linear_index_to_column_index : public thrust::unary_function<T1, T1> {
    T1 rows; // number of rows

    __host__ __device__
    linear_index_to_column_index(T1 rows) : rows(rows) {}

    __host__ __device__
    T1 operator()(T1 i)
    {
        return i / rows;
    }
  };

  sum_operation(const Proxy& proxy, unsigned dimIdx)
   : _proxy(proxy), _size(proxy.size()), dimIdx(dimIdx)
  {
    typename Proxy::dim_t order;
    order[0] = dimIdx;
    for (int i = 1; i < Proxy::dimCount; ++i) {
      order[i] = i - (i <= dimIdx);
    }
    _proxy.reorder(order);
    _size[dimIdx] = 1;
  }

  void apply(tensor_t& output) const {
    thrust::reduce_by_key(
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_column_index<int>(_proxy.size()[0])),
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_column_index<int>(_proxy.size()[0])) + _proxy.count(),
        _proxy.begin(),
        thrust::make_discard_iterator(),
        output.begin()
    );
  }

  inline dim_t size() const {
    return _size;
  }

private:
  Proxy _proxy;
  dim_t _size;
  unsigned dimIdx;
};

template<class T>
struct is_operation<sum_operation<T> > {
  static const bool value = true;
};

template<class Proxy>
typename boost::enable_if<is_proxy<Proxy>,
    typename boost::enable_if_c<Proxy::dimCount >= 1,
      sum_operation<Proxy>
    >::type
>::type
sum(const Proxy& proxy, unsigned dimIdx) {
  assert(dimIdx < Proxy::dimCount);
  return sum_operation<Proxy>(proxy, dimIdx);
}

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
    typename boost::enable_if_c<Tensor::dimCount >= 1,
      sum_operation<proxy<Tensor> >
    >::type
>::type
sum(Tensor& tensor, unsigned dimIdx) {
  assert(dimIdx < Tensor::dimCount);
  return sum(proxy<Tensor>(tensor), dimIdx);
}

}

#endif /* TBBLAS_SUM_HPP_ */
