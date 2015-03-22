/*
 * pooling_method.hpp
 *
 *  Created on: Nov 11, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_POOLING_METHOD_HPP_
#define TBBLAS_DEEPLEARN_POOLING_METHOD_HPP_

#include <capputils/Enumerators.h>

namespace tbblas {

namespace deeplearn {

CapputilsEnumerator(pooling_method, NoPooling, MaxPooling, AvgPooling, StridePooling);

}

}

DefineEnumeratorSerializeTrait(tbblas::deeplearn::pooling_method);


#endif /* TBBLAS_DEEPLEARN_POOLING_METHOD_HPP_ */
