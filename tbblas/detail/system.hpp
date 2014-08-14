/*
 * system.hpp
 *
 *  Created on: Jul 31, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DETAIL_SYSTEM_HPP_
#define TBBLAS_DETAIL_SYSTEM_HPP_

namespace tbblas {

namespace detail {

struct generic_system { };
struct device_system { };

template<bool device>
struct select_system {
  typedef generic_system system;
};

template<>
struct select_system<true> {
  typedef device_system system;
};

}

}

#endif /* TBBLAS_DETAIL_SYSTEM_HPP_ */
