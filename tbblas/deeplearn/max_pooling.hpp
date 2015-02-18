/*
 * max_pooling.hpp
 *
 *  Created on: Feb 16, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_MAX_POOLING_HPP_
#define TBBLAS_DEEPLEARN_MAX_POOLING_HPP_

#include <tbblas/tensor.hpp>

#include <thrust/iterator/transform_iterator.h>
#include <tbblas/sequence_iterator.hpp>

namespace tbblas {

namespace deeplearn {

template<class Tensor>
struct get_max_pooling_switches_expression {

  typedef int difference_type;

  typedef get_max_pooling_switches_expression<Tensor> expression_t;
  typedef difference_type value_t;
  typedef typename Tensor::value_t input_t;
  typedef typename Tensor::dim_t dim_t;

  static const unsigned dimCount = Tensor::dimCount;
  static const bool cuda_enabled = Tensor::cuda_enabled;


  struct switch_operation : public thrust::unary_function<input_t, value_t> {

    switch_operation(input_t* data, const dim_t& in_size, const dim_t& out_size, const dim_t& pooling_size) : data(data), in_size(in_size), out_size(out_size), pooling_size(pooling_size) { }

    __host__ __device__
    inline dim_t to_coords(const difference_type& idx, const dim_t& size) const {
      difference_type index = idx;
      dim_t coord;
      coord[0] = index % size[0];
      for (size_t i = 1; i < dimCount; ++i) {
        coord[i] = (index /= size[i - 1]) % size[i];
      }
      return coord;
    }

    __host__ __device__
    inline difference_type to_linear_index(const dim_t& coord, const dim_t& size) const {
      difference_type index = coord[dimCount - 1];
      for (int i = (int)dimCount - 2; i >= 0; --i) {
        index = index * size[i] + coord[i];
      }
      return index;
    }

    __host__ __device__
    inline value_t operator()(const difference_type& x) const {

      dim_t coord = to_coords(x, out_size);

      sequence_iterator<dim_t> it(seq<dimCount>(0), pooling_size);
      dim_t maxCoord = *it;
      while (it.valid()) {
        if (data[to_linear_index(coord * pooling_size + *it, in_size)] > data[to_linear_index(coord * pooling_size + maxCoord, in_size)]) {
          maxCoord = *it;
        }
        ++it;
      }

      return to_linear_index(maxCoord, pooling_size);
    }

  private:
    input_t* data;
    dim_t in_size, out_size, pooling_size;
  };

  typedef thrust::counting_iterator<difference_type> CountingIterator;
  typedef thrust::transform_iterator<switch_operation, CountingIterator> const_iterator;

  get_max_pooling_switches_expression(Tensor& tensor, const dim_t& pooling_size)
   : tensor(tensor), pooling_size(pooling_size), outSize(tensor.size() / pooling_size), outFullsize(tensor.fullsize() / pooling_size) { }

  inline const_iterator begin() const {
    return thrust::make_transform_iterator(
        CountingIterator(0),
        switch_operation(thrust::raw_pointer_cast(tensor.data().data()), tensor.size(), outSize, pooling_size));
  }

  inline const_iterator end() const {
    return begin() + count();
  }

  inline dim_t size() const {
    return outSize;
  }

  inline dim_t fullsize() const {
    return outFullsize;
  }

  inline size_t count() const {
    return outSize.prod();
  }

private:
  Tensor& tensor;
  dim_t pooling_size, outSize, outFullsize;
};

}

template<class T>
struct is_expression<tbblas::deeplearn::get_max_pooling_switches_expression<T> > {
  static const bool value = true;
};

namespace deeplearn {

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
  get_max_pooling_switches_expression<Tensor>
>::type
get_max_pooling_switches(Tensor& tensor, const sequence<int, Tensor::dimCount>& pooling_size)
{
  assert(tensor.size() % pooling_size == seq<Tensor::dimCount>(0));
  return get_max_pooling_switches_expression<Tensor>(tensor, pooling_size);
}

/*** POOLING ***/

template<class Tensor1, class Tensor2>
struct max_pooling_expression {

  typedef max_pooling_expression<Tensor1, Tensor2> expression_t;
  typedef typename Tensor1::value_t value_t;
  typedef typename Tensor2::value_t input_t;
  typedef typename Tensor1::dim_t dim_t;

  static const unsigned dimCount = Tensor1::dimCount;
  static const bool cuda_enabled = Tensor1::cuda_enabled;

  typedef int difference_type;

  struct max_pooling_operation : public thrust::unary_function<difference_type, value_t> {

    max_pooling_operation(value_t* data, input_t* switches, const dim_t& in_size, const dim_t& out_size, const dim_t& pooling_size)
      : data(data), switches(switches), in_size(in_size), out_size(out_size), pooling_size(pooling_size) { }

    __host__ __device__
    inline dim_t to_coords(const difference_type& idx, const dim_t& size) const {
      difference_type index = idx;
      dim_t coord;
      coord[0] = index % size[0];
      for (size_t i = 1; i < dimCount; ++i) {
        coord[i] = (index /= size[i - 1]) % size[i];
      }
      return coord;
    }

    __host__ __device__
    inline difference_type to_linear_index(const dim_t& coord, const dim_t& size) const {
      difference_type index = coord[dimCount - 1];
      for (int i = (int)dimCount - 2; i >= 0; --i) {
        index = index * size[i] + coord[i];
      }
      return index;
    }

    __host__ __device__
    inline value_t operator()(const difference_type& x) const {
      return data[to_linear_index(to_coords(x, out_size) * pooling_size + to_coords(switches[x], pooling_size), in_size)];
    }

  private:
    value_t* data;
    input_t* switches;
    dim_t in_size, out_size, pooling_size;
  };

  typedef thrust::counting_iterator<difference_type> CountingIterator;
  typedef thrust::transform_iterator<max_pooling_operation, CountingIterator> const_iterator;

  max_pooling_expression(Tensor1& tensor, Tensor2& switches)
   : tensor(tensor), switches(switches), pooling_size(tensor.size() / switches.size()), outSize(switches.size()), outFullsize(tensor.fullsize() / pooling_size) { }

  inline const_iterator begin() const {
    return thrust::make_transform_iterator(
        CountingIterator(0),
        max_pooling_operation(thrust::raw_pointer_cast(tensor.data().data()),
            thrust::raw_pointer_cast(switches.data().data()),
            tensor.size(), outSize, pooling_size));
  }

  inline const_iterator end() const {
    return begin() + count();
  }

  inline dim_t size() const {
    return outSize;
  }

  inline dim_t fullsize() const {
    return outFullsize;
  }

  inline size_t count() const {
    return outSize.prod();
  }

private:
  Tensor1& tensor;
  Tensor2& switches;
  dim_t pooling_size, outSize, outFullsize;
};

}

template<class T, class U>
struct is_expression<tbblas::deeplearn::max_pooling_expression<T, U> > {
  static const bool value = true;
};

namespace deeplearn {

template<class Tensor1, class Tensor2>
typename boost::enable_if<is_tensor<Tensor1>,
  typename boost::enable_if<is_tensor<Tensor2>,
    typename boost::enable_if_c<Tensor1::dimCount == Tensor2::dimCount,
      max_pooling_expression<Tensor1, Tensor2>
    >::type
  >::type
>::type
max_pooling(Tensor1& tensor, Tensor2& switches, const typename Tensor1::dim_t& pooling_size)
{
  assert(tensor.size() == switches.size() * pooling_size);
  return max_pooling_expression<Tensor1, Tensor2>(tensor, switches);
}

/*** UNPOOLING ***/

template<class Tensor1, class Tensor2>
struct unpooling_expression {

  typedef unpooling_expression<Tensor1, Tensor2> expression_t;
  typedef typename Tensor1::value_t value_t;
  typedef typename Tensor2::value_t input_t;
  typedef typename Tensor1::dim_t dim_t;

  static const unsigned dimCount = Tensor1::dimCount;
  static const bool cuda_enabled = Tensor1::cuda_enabled;

  typedef int difference_type;

  struct unpooling_operation : public thrust::unary_function<difference_type, value_t> {

    unpooling_operation(value_t* data, input_t* switches, const dim_t& in_size, const dim_t& out_size, const dim_t& pooling_size)
      : data(data), switches(switches), in_size(in_size), out_size(out_size), pooling_size(pooling_size) { }

    __host__ __device__
    inline dim_t to_coords(const difference_type& idx, const dim_t& size) const {
      difference_type index = idx;
      dim_t coord;
      coord[0] = index % size[0];
      for (size_t i = 1; i < dimCount; ++i) {
        coord[i] = (index /= size[i - 1]) % size[i];
      }
      return coord;
    }

    __host__ __device__
    inline difference_type to_linear_index(const dim_t& coord, const dim_t& size) const {
      difference_type index = coord[dimCount - 1];
      for (int i = (int)dimCount - 2; i >= 0; --i) {
        index = index * size[i] + coord[i];
      }
      return index;
    }

    __host__ __device__
    inline value_t operator()(const difference_type& x) const {
      dim_t coord = to_coords(x, out_size);
      dim_t scoord = coord / pooling_size;
      dim_t bcoord = coord - scoord * pooling_size;

      if (switches[to_linear_index(scoord, in_size)] == to_linear_index(bcoord, pooling_size)) {
        return data[to_linear_index(scoord, in_size)];
      } else {
        return 0;
      }
    }

  private:
    value_t* data;
    input_t* switches;
    dim_t in_size, out_size, pooling_size;
  };

  typedef thrust::counting_iterator<difference_type> CountingIterator;
  typedef thrust::transform_iterator<unpooling_operation, CountingIterator> const_iterator;

  unpooling_expression(Tensor1& tensor, Tensor2& switches, const dim_t& pooling_size)
   : tensor(tensor), switches(switches), pooling_size(pooling_size), outSize(tensor.size() * pooling_size), outFullsize(tensor.fullsize() * pooling_size) { }

  inline const_iterator begin() const {
    return thrust::make_transform_iterator(
        CountingIterator(0),
        unpooling_operation(thrust::raw_pointer_cast(tensor.data().data()),
            thrust::raw_pointer_cast(switches.data().data()),
            tensor.size(), outSize, pooling_size));
  }

  inline const_iterator end() const {
    return begin() + count();
  }

  inline dim_t size() const {
    return outSize;
  }

  inline dim_t fullsize() const {
    return outFullsize;
  }

  inline size_t count() const {
    return outSize.prod();
  }

private:
  Tensor1& tensor;
  Tensor2& switches;
  dim_t pooling_size, outSize, outFullsize;
};

}

template<class T, class U>
struct is_expression<tbblas::deeplearn::unpooling_expression<T, U> > {
  static const bool value = true;
};

namespace deeplearn {

template<class Tensor1, class Tensor2>
typename boost::enable_if<is_tensor<Tensor1>,
  typename boost::enable_if<is_tensor<Tensor2>,
    typename boost::enable_if_c<Tensor1::dimCount == Tensor2::dimCount,
      unpooling_expression<Tensor1, Tensor2>
    >::type
  >::type
>::type
unpooling(Tensor1& tensor, Tensor2& switches, const typename Tensor1::dim_t& pooling_size)
{
  assert(tensor.size() == switches.size());
  return unpooling_expression<Tensor1, Tensor2>(tensor, switches, pooling_size);
}

}

}

#endif /* TBBLAS_DEEPLEARN_MAX_POOLING_HPP_ */
