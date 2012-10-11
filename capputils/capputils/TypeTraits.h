/*
 * TypeTraits.h
 *
 *  Created on: Oct 10, 2012
 *      Author: tombr
 */

#ifndef CAPPUTILS_TYPETRAITS_H_
#define CAPPUTLIS_TYPETRAITS_H_

namespace capputils {

template<class T>
struct is_enumerator {
  static const bool value = false;
};

}

#endif /* TYPETRAITS_H_ */
