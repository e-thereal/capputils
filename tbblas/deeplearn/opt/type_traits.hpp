/*
 * type_traits.hpp
 *
 *  Created on: Apr 15, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_OPT_TYPE_TRAITS_HPP_
#define TBBLAS_DEEPLEARN_OPT_TYPE_TRAITS_HPP_

namespace tbblas {

namespace deeplearn {

namespace opt {

template<class T>
struct is_trainer {
  static const bool value = false;
};

}

}

}

#endif /* TBBLAS_DEEPLEARN_OPT_TYPE_TRAITS_HPP_ */
