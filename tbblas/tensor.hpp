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
#include <boost/type_traits/is_same.hpp>

#include <tbblas/type_traits.hpp>
#include <tbblas/sequence.hpp>

namespace tbblas {

template<class T, unsigned dim, bool device>
class tensor;

template<class Tensor>
struct proxy;

template<class T, unsigned dim, bool device>
proxy<tensor<T, dim, device> > subrange(tensor<T, dim, device>& t,
    const sequence<unsigned, dim>& start, const sequence<unsigned, dim>& size);

template<class T, unsigned dim, bool device = true>
class tensor {
public:

  typedef tensor<T, dim, device> tensor_t;

  typedef typename vector_type<T, device>::vector_t data_t;
  typedef T value_t;
  typedef size_t dim_t[dim];

  const static unsigned dimCount = dim;
  const static bool cuda_enabled = device;

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

  template<class Operation>
  inline
  tensor(const Operation& op, typename boost::enable_if<is_operation<Operation>,
      typename boost::enable_if<boost::is_same<typename Operation::tensor_t, tensor_t>, int>::type>::type = 0)
  {
    const dim_t& size = op.size();
    for (unsigned i = 0; i < dimCount; ++i) {
      _size[i] = size[i];
    }
    _data = boost::shared_ptr<data_t>(new data_t(count()));
    op.apply(*this);
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

  boost::shared_ptr<data_t> shared_data() {
    return _data;
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

  typename data_t::reference operator[](const sequence<unsigned, dim>& index) {
    size_t idx = index[dim-1];
    for (int i = dim - 2; i >= 0; --i) {
      idx = idx * _size[i] + index[i];
    }
    return data()[idx];
  }

  typename data_t::reference operator[](const sequence<int, dim>& index) {
    size_t idx = index[dim-1];
    for (int i = dim - 2; i >= 0; --i) {
      idx = idx * _size[i] + index[i];
    }
    return data()[idx];
  }

  const proxy<tensor_t> operator[](const std::pair<sequence<unsigned, dim>, sequence<unsigned, dim> >& pair) {
    return subrange(*this, pair.first, pair.second);
  }

  const proxy<tensor_t> operator[](const std::pair<sequence<int, dim>, sequence<int, dim> >& pair) {
    return subrange(*this, pair.first, pair.second);
  }

  /*** apply operations ***/

  template<class T2, bool device2>
  inline tensor_t& operator=(const tensor<T2, dim, device2>& tensor) {
    const dim_t& size = tensor.size();
    bool realloc = false;

    for (unsigned i = 0; i < dim; ++i) {
      if (size[i] != _size[i]) {
        realloc = true;
        _size[i] = size[i];
      }
    }

    if (realloc) {
      _data = boost::shared_ptr<data_t>(new data_t(count()));
    }
    thrust::copy(tensor.begin(), tensor.end(), begin());

    return *this;
  }

  template<class Operation>
  inline typename boost::enable_if<is_operation<Operation>,
    typename boost::enable_if<boost::is_same<typename Operation::tensor_t, tensor_t>,
      tensor_t
    >::type
  >::type&
  operator=(const Operation& op)
  {
    const dim_t& size = op.size();
    bool realloc = false;
    for (unsigned i = 0; i < dimCount; ++i) {
      if (size[i] != _size[i]) {
        realloc = true;
        _size[i] = size[i];
      }
    }

    if (realloc) {
      _data = boost::shared_ptr<data_t>(new data_t(count()));
    }
    op.apply(*this);
    return *this;
  }

  template<class Expression>
  inline typename boost::enable_if<is_expression<Expression>,
    typename boost::enable_if_c<Expression::dimCount == dimCount,
      typename boost::disable_if<is_operation<Expression>,
        tensor_t
      >::type
    >::type
  >::type&
  operator=(const Expression& expr)
  {
    const dim_t& size = expr.size();
    bool realloc = false;
    for (unsigned i = 0; i < dimCount; ++i) {
      if (size[i] != _size[i]) {
        realloc = true;
        _size[i] = size[i];
      }
    }

    if (realloc) {
      _data = boost::shared_ptr<data_t>(new data_t(count()));
    }
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

// Include default headers

#include <tbblas/proxy.hpp>
#include <tbblas/plus2.hpp>
#include <tbblas/minus.hpp>
#include <tbblas/multiplies.hpp>
#include <tbblas/divides.hpp>

#endif /* TBBLAS_TENSOR_HPP_ */