/*
 * sparsity_method.hpp
 *
 *  Created on: Jul 2, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_SPARSITY_METHOD_HPP_
#define TBBLAS_DEEPLEARN_SPARSITY_METHOD_HPP_

#include <capputils/Enumerators.h>

namespace tbblas {

namespace deeplearn {

CapputilsEnumerator(sparsity_method, NoSparsity, WeightsAndBias, OnlyBias, OnlySharedBias);

}

}

DefineEnumeratorSerializeTrait(tbblas::deeplearn::sparsity_method);

#endif /* TBBLAS_DEEPLEARN_SPARSITY_METHOD_HPP_ */
