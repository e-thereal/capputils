/*
 * activation_function.hpp
 *
 *  Created on: Aug 13, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_ACTIVATIONFUNCTION_H_
#define TBBLAS_DEEPLEARN_ACTIVATIONFUNCTION_H_

#include <capputils/Enumerators.h>

namespace tbblas {

namespace deeplearn {

CapputilsEnumerator(activation_function, Sigmoid, ReLU, Softmax);

}

}

DefineEnumeratorSerializeTrait(tbblas::deeplearn::activation_function);

#endif /* TBBLAS_DEEPLEARN_ACTIVATIONFUNCTION_H_ */
