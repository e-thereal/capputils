/*
 * objective_function.hpp
 *
 *  Created on: Dec 6, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_OBJECTIVE_FUNCTION_HPP_
#define TBBLAS_DEEPLEARN_OBJECTIVE_FUNCTION_HPP_

#include <capputils/Enumerators.h>

namespace tbblas {

namespace deeplearn {

CapputilsEnumerator(objective_function, SSD, SenSpe, DSC, DSC2);

}

}

DefineEnumeratorSerializeTrait(tbblas::deeplearn::objective_function);

#endif /* TBBLAS_DEEPLEARN_OBJECTIVE_FUNCTION_HPP_ */
