/*
 * avg_pooling.hpp
 *
 *  Created on: Mar 20, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_AVG_POOLING_HPP_
#define TBBLAS_DEEPLEARN_AVG_POOLING_HPP_

#include <tbblas/tensor.hpp>

#include <thrust/iterator/transform_iterator.h>
#include <tbblas/sequence_iterator.hpp>

namespace tbblas {

namespace deeplearn {

template<class Tensor>
struct avg_pooling_expression {

  typedef int difference_type;

  typedef avg_pooling_expression<Tensor> expression_t;
  typedef typename Tensor::value_t value_t;
  typedef typename Tensor::dim_t dim_t;

  static const unsigned dimCount = Tensor::dimCount;
  static const bool cuda_enabled = Tensor::cuda_enabled;


  struct avg_pooling_operation : public thrust::unary_function<difference_type, value_t> {

    avg_pooling_operation(value_t* data, const dim_t& in_size, const dim_t& out_size, const dim_t& pooling_size) : data(data), in_size(in_size), out_size(out_size), pooling_size(pooling_size) { }

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
      value_t avg = 0;
      while (it.valid()) {
        avg += data[to_linear_index(coord * pooling_size + *it, in_size)];
        ++it;
      }

      return avg / it.count();
    }

  private:
    value_t* data;
    dim_t in_size, out_size, pooling_size;
  };

  typedef thrust::counting_iterator<difference_type> CountingIterator;
  typedef thrust::transform_iterator<avg_pooling_operation, CountingIterator> const_iterator;

  avg_pooling_expression(Tensor& tensor, const dim_t& pooling_size)
   : tensor(tensor), pooling_size(pooling_size), outSize(tensor.size() / pooling_size), outFullsize(tensor.fullsize() / pooling_size) { }

  inline const_iterator begin() const {
    return thrust::make_transform_iterator(
        CountingIterator(0),
        avg_pooling_operation(thrust::raw_pointer_cast(tensor.data().data()), tensor.size(), outSize, pooling_size));
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
struct is_expression<tbblas::deeplearn::avg_pooling_expression<T> > {
  static const bool value = true;
};

namespace deeplearn {

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
  avg_pooling_expression<Tensor>
>::type
avg_pooling(Tensor& tensor, const sequence<int, Tensor::dimCount>& pooling_size)
{
  assert(tensor.size() % pooling_size == seq<Tensor::dimCount>(0));
  return avg_pooling_expression<Tensor>(tensor, pooling_size);
}

/*** UNPOOLING ***/

template<class Tensor>
struct avg_unpooling_expression {

  typedef avg_unpooling_expression<Tensor> expression_t;
  typedef typename Tensor::value_t value_t;
  typedef typename Tensor::dim_t dim_t;

  static const unsigned dimCount = Tensor::dimCount;
  static const bool cuda_enabled = Tensor::cuda_enabled;

  typedef int difference_type;

  struct unpooling_operation : public thrust::unary_function<difference_type, value_t> {

    unpooling_operation(value_t* data, const dim_t& in_size, const dim_t& out_size, const dim_t& pooling_size)
      : data(data), in_size(in_size), out_size(out_size), pooling_size(pooling_size) { }

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

      return data[to_linear_index(scoord, in_size)];
    }

  private:
    value_t* data;
    dim_t in_size, out_size, pooling_size;
  };

  typedef thrust::counting_iterator<difference_type> CountingIterator;
  typedef thrust::transform_iterator<unpooling_operation, CountingIterator> const_iterator;

  avg_unpooling_expression(Tensor& tensor, const dim_t& pooling_size)
   : tensor(tensor), pooling_size(pooling_size), outSize(tensor.size() * pooling_size), outFullsize(tensor.fullsize() * pooling_size) { }

  inline const_iterator begin() const {
    return thrust::make_transform_iterator(
        CountingIterator(0),
        unpooling_operation(thrust::raw_pointer_cast(tensor.data().data()),
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
  Tensor& tensor;
  dim_t pooling_size, outSize, outFullsize;
};

}

template<class T>
struct is_expression<tbblas::deeplearn::avg_unpooling_expression<T> > {
  static const bool value = true;
};

namespace deeplearn {

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
  avg_unpooling_expression<Tensor>
>::type
unpooling(Tensor& tensor, const typename Tensor::dim_t& pooling_size)
{
  return avg_unpooling_expression<Tensor>(tensor, pooling_size);
}

}

}

#endif /* TBBLAS_DEEPLEARN_AVG_POOLING_HPP_ */
