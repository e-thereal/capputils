/*
 * expand.hpp
 *
 *  Created on: Oct 3, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_EXPAND_HPP_
#define TBBLAS_EXPAND_HPP_

#include <tbblas/sequence.hpp>

#include <tbblas/tensor.hpp>
#include <boost/static_assert.hpp>

#include <boost/utility/enable_if.hpp>

#include <cassert>

namespace tbblas {

template<class Tensor>
struct fftexpand_expression {

  typedef fftexpand_expression<Tensor> expression_t;
  typedef typename Tensor::value_t value_t;
  typedef typename Tensor::dim_t dim_t;

  static const int dimCount = Tensor::dimCount;
  static const bool cuda_enabled = Tensor::cuda_enabled;

  typedef int difference_type;

  // Maps an expanded index to a memory index
  struct index_functor : public thrust::unary_function<difference_type,difference_type> {

    dim_t inSize;
    difference_type outWidth;   ///< width of memory

    index_functor(dim_t inSize, difference_type outWidth)
     : inSize(inSize), outWidth(outWidth) { }

    __host__ __device__
    difference_type operator()(difference_type idx) const
    {
      size_t count1 = 0, count2 = 1;

      difference_type x = idx % inSize[0];
      if (x >= outWidth) {
        for (unsigned k = 0; k < dimCount; ++k) {
          count1 += count2;
          count2 *= inSize[k];
          if (idx < count2) {
            idx = count1 + count2 - idx - 1;
            x = idx % inSize[0];
            break;
          }
        }
      }
      return idx / inSize[0] * outWidth + x;
    }
  };

  typedef thrust::counting_iterator<difference_type>                  CountingIterator;
  typedef thrust::transform_iterator<index_functor, CountingIterator> TransformIterator;
  typedef thrust::permutation_iterator<typename Tensor::const_iterator,TransformIterator> PermutationIterator;
  typedef PermutationIterator const_iterator;

  fftexpand_expression(const Tensor& tensor) : tensor(tensor) { }

  inline const_iterator begin() const {
    index_functor functor(tensor.full_size(), tensor.size()[0]);
    CountingIterator counting(0);
    TransformIterator transform(counting, functor);
    PermutationIterator permu(tensor.begin(), transform);
    return permu;
//    return PermutationIterator(first, TransformIterator(CountingIterator(0), subrange_functor(_size, _pitch)));
  }

  inline const_iterator end() const {
    return begin() + count();
  }

  inline const dim_t& size() const {
    return tensor.full_size();
  }

  inline size_t count() const {
    size_t count = 1;
    for (unsigned i = 0; i < dimCount; ++i)
      count *= tensor.full_size()[i];
    return count;
  }

private:
  const Tensor& tensor;
};

template<class T>
struct is_expression<fftexpand_expression<T> > {
  static const bool value = true;
};

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
  fftexpand_expression<Tensor>
>::type
fftexpand(const Tensor& tensor)
{
  return fftexpand_expression<Tensor>(tensor);
}

}

#endif /* TBBLAS_EXPAND_HPP_ */
