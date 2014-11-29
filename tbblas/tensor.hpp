/*
 * tensor.hpp
 *
 *  Created on: Sep 7, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_TENSOR_HPP_
#define TBBLAS_TENSOR_HPP_

#include <tbblas/detail/copy.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_float.hpp>
#include <boost/static_assert.hpp>

#include <tbblas/tbblas.hpp>
#include <tbblas/type_traits.hpp>
#include <tbblas/sequence.hpp>
#include <tbblas/fill.hpp>
#include <tbblas/context.hpp>

#include <iostream>

namespace tbblas {

template<class T, unsigned dim, bool device>
class tensor;

template<class Tensor>
struct proxy;

template<class T, unsigned dim, bool device>
proxy<tensor<T, dim, device> > subrange(tensor<T, dim, device>& t,
    const sequence<unsigned, dim>& start, const sequence<unsigned, dim>& size);

template<class T, unsigned dim, bool device = false>
class tensor {
public:

  typedef tensor<T, dim, device> tensor_t;

  typedef typename vector_type<T, device>::vector_t data_t;
  typedef T value_t;
  typedef sequence<int, dim> dim_t;

  const static unsigned dimCount = dim;
  const static bool cuda_enabled = device;

public:
  typedef typename data_t::iterator iterator;
  typedef typename data_t::const_iterator const_iterator;

protected:
  boost::shared_ptr<data_t> _data;
  dim_t _size , _fullsize; // fullsize is not used because this member is not delegated consistently
  std::string name;

public:
  tensor(size_t x1 = 1, size_t x2 = 1, size_t x3 = 1, size_t x4 = 1) {
    TBBLAS_ALLOC_WARNING

    const size_t size[] = {x1, x2, x3, x4};
    size_t count = 1;

    for (unsigned i = 0; i < dim; ++i) {
      _size[i] = size[i];
      count *= size[i];
    }
    _fullsize = _size;

    _data = boost::shared_ptr<data_t>(new data_t(count));
  }

  tensor(const dim_t& size) {
    TBBLAS_ALLOC_WARNING

    size_t count = 1;
    for (unsigned i = 0; i < dim; ++i) {
      _size[i] = size[i];
      count *= size[i];
    }
    _fullsize = _size;

    _data = boost::shared_ptr<data_t>(new data_t(count));
  }

  tensor(const tensor_t& tensor) {
    TBBLAS_ALLOC_WARNING

    const dim_t& size = tensor.size();

    size_t count = 1;
    for (unsigned i = 0; i < dim; ++i) {
      _size[i] = size[i];
      count *= size[i];
    }
    _fullsize = tensor.fullsize();

    _data = boost::shared_ptr<data_t>(new data_t(count));
    tbblas::detail::copy(typename tbblas::detail::select_system<device>::system(), tensor.begin(), tensor.end(), begin());
  }

  template<bool device2>
  tensor(const tensor<T, dim, device2>& tensor, typename boost::enable_if_c<device != device2, int>::type = 0) {
    TBBLAS_ALLOC_WARNING

    const dim_t& size = tensor.size();
    size_t count = 1;
    for (unsigned i = 0; i < dim; ++i) {
      _size[i] = size[i];
      count *= size[i];
    }
    _fullsize = tensor.fullsize();
    _data = boost::shared_ptr<data_t>(new data_t(count));

    cudaMemcpyAsync(thrust::raw_pointer_cast(_data->data()),
        thrust::raw_pointer_cast(tensor.data().data()),
        count * sizeof(T),
        (device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost),
        tbblas::context::get().stream);
  }

  template<class T2, bool device2>
  tensor(const tensor<T2, dim, device2>& tensor) {
    TBBLAS_ALLOC_WARNING

    const dim_t& size = tensor.size();

    size_t count = 1;
    for (unsigned i = 0; i < dim; ++i) {
      _size[i] = size[i];
      count *= size[i];
    }
    _fullsize = tensor.fullsize();

    _data = boost::shared_ptr<data_t>(new data_t(count));
    tbblas::detail::copy(typename tbblas::detail::select_system<device && device2>::system(), tensor.begin(), tensor.end(), begin());
  }

  template<class Operation>
  inline
  tensor(const Operation& op, typename boost::enable_if<is_operation<Operation>,
      typename boost::enable_if<boost::is_same<typename Operation::tensor_t, tensor_t>, int>::type>::type = 0)
  {
    TBBLAS_ALLOC_WARNING
    const dim_t& size = op.size();
    for (unsigned i = 0; i < dimCount; ++i) {
      _size[i] = size[i];
    }
    _fullsize = op.fullsize();

    _data = boost::shared_ptr<data_t>(new data_t(count()));
    op.apply(*this);
  }

  template<class Expression>
  tensor(const Expression& expr, typename boost::enable_if<is_expression<Expression>,
      typename boost::enable_if_c<Expression::dimCount == dimCount, int>::type>::type = 0)
  {
    TBBLAS_ALLOC_WARNING

    const dim_t& size = expr.size();
    for (unsigned i = 0; i < dimCount; ++i) {
      _size[i] = size[i];
    }
    _fullsize = expr.fullsize();

    _data = boost::shared_ptr<data_t>(new data_t(count()));
    tbblas::detail::copy(typename tbblas::detail::select_system<device && Expression::cuda_enabled>::system(), expr.begin(), expr.end(), begin());
  }

  virtual ~tensor() {
    TBBLAS_FREE_MESSAGE
  }

public:
  void set_name(const std::string& name) {
    this->name = name;
  }

  inline dim_t size() const {
    return _size;
  }

  inline void resize(const dim_t& size) {
    // README: use 'void resize(const dim_t& size, const dim_t fullsize)' instead for complex value types!
    BOOST_STATIC_ASSERT(!tbblas::is_complex<value_t>::value);
    bool realloc = false;

    for (unsigned i = 0; i < dim; ++i) {
      if (size[i] != _size[i]) {
        realloc = true;
        _size[i] = size[i];
      }
    }
    _fullsize = size;

    if (realloc) {
      TBBLAS_ALLOC_WARNING
      _data = boost::shared_ptr<data_t>(new data_t(count()));
    }
  }

  inline void resize(const dim_t& size, const dim_t fullsize) {
    bool realloc = false;

    for (unsigned i = 0; i < dim; ++i) {
      if (size[i] != _size[i]) {
        realloc = true;
        _size[i] = size[i];
      }
    }
    _fullsize = fullsize;

    if (realloc) {
      TBBLAS_ALLOC_WARNING
      _data = boost::shared_ptr<data_t>(new data_t(count()));
    }
  }

  inline dim_t fullsize() const {
    return _fullsize;
  }

  void set_fullsize(const dim_t& size) {
    _fullsize = size;
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
    for (int i = (int)dim - 2; i >= 0; --i) {
      idx = idx * _size[i] + index[i];
    }
    return data()[idx];
  }

  typename data_t::reference operator[](const sequence<int, dim>& index) {
    size_t idx = index[dim-1];
    for (int i = (int)dim - 2; i >= 0; --i) {
      idx = idx * _size[i] + index[i];
    }
    return data()[idx];
  }

  // returning const proxy in order to avoid copy constructors when assigning values to returned proxy
  const proxy<tensor_t> operator[](const std::pair<sequence<unsigned, dim>, sequence<unsigned, dim> >& pair) {
    return subrange(*this, pair.first, pair.second);
  }

  const proxy<tensor_t> operator[](const std::pair<sequence<int, dim>, sequence<int, dim> >& pair) {
    return subrange(*this, pair.first, pair.second);
  }

  const proxy<tensor_t> operator[](const std::pair<std::pair<sequence<int, dim>, sequence<int, dim> >, sequence<int, dim> >& pair) {
    return slice(*this, pair.first.first, pair.first.second, pair.second);
  }

  /*** apply operations ***/

  inline tensor_t& operator=(const tensor_t& tensor) {
    resize(tensor.size(), tensor.fullsize());
    tbblas::detail::copy(typename tbblas::detail::select_system<device>::system(), tensor.begin(), tensor.end(), begin());
    return *this;
  }


  template<bool device2>
  inline typename boost::enable_if_c<device != device2, tensor>::type&
  operator=(const tensor<T, dim, device2>& t) {
    resize(t.size(), t.fullsize());

    cudaMemcpyAsync(thrust::raw_pointer_cast(_data->data()),
        thrust::raw_pointer_cast(t.data().data()),
        count() * sizeof(T),
        (device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost),
        tbblas::context::get().stream);
    return *this;
  }

  template<class T2, bool device2>
  inline tensor_t& operator=(const tensor<T2, dim, device2>& tensor) {
    resize(tensor.size(), tensor.fullsize());
    tbblas::detail::copy(typename tbblas::detail::select_system<device && device2>::system(), tensor.begin(), tensor.end(), begin());
    return *this;
  }

  template<class Operation>
  inline typename boost::enable_if<is_operation<Operation>,
    typename boost::enable_if<boost::is_same<typename Operation::tensor_t, tensor_t>,
      tensor_t
    >::type
  >::type&
  operator=(const Operation& op) {
    resize(op.size(), op.fullsize());
    op.apply(*this);
    return *this;
  }

  template<class Operation>
  inline typename boost::enable_if<is_inc_operation<Operation>,
    typename boost::enable_if<boost::is_same<typename Operation::tensor_t, tensor_t>,
      tensor_t
    >::type
  >::type&
  operator+=(const Operation& op) {
    for (int i = 0; i < dimCount; ++i) {
      assert(op.size()[i] == size()[i]);
      assert(op.fullsize()[i] == fullsize()[i]);
    }
    op.apply_inc(*this);
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
  operator=(const Expression& expr) {
    resize(expr.size(), expr.fullsize());
    tbblas::detail::copy(typename tbblas::detail::select_system<device && Expression::cuda_enabled>::system(), expr.begin(), expr.end(), begin());
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
  operator+=(const Expression& expr) {
    *this = *this + expr;
    return *this;
  }

  proxy_filler<proxy<tensor_t> > operator=(const value_t& value) {
    return subrange(*this, seq<dimCount>(0), size()) = value;
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
#include <tbblas/arithmetic.hpp>
#include <tbblas/comparisons.hpp>

#endif /* TBBLAS_TENSOR_HPP_ */
