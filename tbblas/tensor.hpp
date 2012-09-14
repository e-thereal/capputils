/*
 * tensor.hpp
 *
 *  Created on: Sep 7, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_TENSOR_HPP_
#define TBBLAS_TENSOR_HPP_

#include <thrust/functional.h>

#include <tbblas/forward_reverse_iterator.hpp>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/copy.h>
#include <boost/shared_ptr.hpp>
#include <boost/utility/enable_if.hpp>

#include <tbblas/type_traits.hpp>

namespace tbblas {

template<class Tensor, class Operation>
void apply_operation2(Tensor& tensor, const Operation& operation) {
  assert(0);
}

template<class T, unsigned dim, bool device = true>
class tensor {
public:

  typedef tensor<T, dim, device> tensor_t;

  typedef typename vector_type<T, device>::vector_t data_t;
  typedef T value_t;
  typedef size_t dim_t[dim];

  const static unsigned dimCount = dim;

public:
  typedef typename data_t::iterator iterator;
  typedef typename data_t::const_iterator const_iterator;

protected:
  boost::shared_ptr<data_t> _data;
  dim_t _size;

public:
  tensor(size_t x1 = 1, size_t x2 = 1, size_t x3 = 1, size_t x4 = 1) {
    const size_t size[] = {x1, x2, x3, x4};
    size_t count = 1;

    for (unsigned i = 0; i < dim; ++i) {
      _size[i] = size[i];
      count *= size[i];
    }

    _data = boost::shared_ptr<data_t>(new data_t(count));
  }

  tensor(const dim_t& size) {
    size_t count = 1;
    for (unsigned i = 0; i < dim; ++i) {
      _size[i] = size[i];
      count *= size[i];
    }
    _data = boost::shared_ptr<data_t>(new data_t(count));
  }

  template<class T2, bool device2>
  tensor(const tensor<T2, dim, device2>& tensor) {
    const dim_t& size = tensor.size();

    size_t count = 1;
    for (unsigned i = 0; i < dim; ++i) {
      _size[i] = size[i];
      count *= size[i];
    }
    _data = boost::shared_ptr<data_t>(new data_t(count));
    thrust::copy(tensor.begin(), tensor.end(), begin());
  }

  template<class Expression>
  tensor(const Expression& expr, typename boost::enable_if<is_expression<Expression>,
      typename boost::enable_if_c<Expression::dimCount == dimCount, int>::type>::type = 0)
  {
    const dim_t& size = expr.size();
    for (unsigned i = 0; i < dimCount; ++i) {
      _size[i] = size[i];
    }
    _data = boost::shared_ptr<data_t>(new data_t(count()));
    thrust::copy(expr.begin(), expr.end(), begin());
  }

  virtual ~tensor() { }

public:
  inline const dim_t& size() const {
    return _size;
  }

  inline size_t count() const {
    size_t count = 1;
    for (unsigned i = 0; i < dim; ++i) {
      count *= _size[i];
    }
    return count;
  }

  inline data_t& data() {
    return *_data;
  }

  inline const data_t& data() const {
    return *_data;
  }

  inline iterator begin() {
    return _data->begin();
  }

  inline iterator end() {
    return _data->end();
  }

  inline const_iterator begin() const {
    return _data->begin();
  }

  inline const_iterator end() const {
    return _data->end();
  }

  /*** apply operations ***/

  template<class T2, bool device2>
  inline tensor_t& operator=(const tensor<T2, dim, device2>& tensor) {
    const dim_t& size = tensor.size();
    bool realloc = false;

    for (unsigned i = 0; i < dim; ++i) {
      if (size[i] != _size[1]) {
        realloc = true;
        _size[i] = size[i];
      }
    }

    if (realloc)
      _data = boost::shared_ptr<data_t>(new data_t(count()));
    thrust::copy(tensor.begin(), tensor.end(), begin());

    return *this;
  }

  template<class Operation>
  inline typename boost::enable_if<is_operation<Operation>,
      typename boost::enable_if_c<Operation::dimCount == dimCount, tensor_t&>::type>::type
  operator=(const Operation& op) {
    apply_operation2(*this, op);
    return *this;
  }

  template<class Expression>
  inline typename boost::enable_if<is_expression<Expression>,
      typename boost::enable_if_c<Expression::dimCount == dimCount, tensor_t&>::type>::type
  operator=(const Expression& expr) {
    const dim_t& size = expr.size();
    bool realloc = false;
    for (unsigned i = 0; i < dimCount; ++i) {
      if (size[i] != _size[1]) {
        realloc = true;
        _size[i] = size[i];
      }
    }

    if (realloc)
      _data = boost::shared_ptr<data_t>(new data_t(count()));
    thrust::copy(expr.begin(), expr.end(), begin());

    return *this;
  }
};

template<class T, unsigned dim, bool device>
struct is_tensor<tensor<T, dim, device> > {
  static const bool value = true;
};

template<class T, unsigned dim, bool device>
struct is_expression<tensor<T, dim, device> > {
  static const bool value = true;
};

}

#endif /* TBBLAS_TENSOR_HPP_ */
