/*
 * UnitType.h
 *
 *  Created on: Nov 21, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_UNITTYPE_H_
#define TBBLAS_DEEPLEARN_UNITTYPE_H_

#include <capputils/Enumerators.h>

namespace tbblas {

namespace deeplearn {

CapputilsEnumerator(unit_type, Bernoulli, Gaussian, MyReLU, ReLU, ReLU1, ReLU2, ReLU4, ReLU8);

}

}

DefineEnumeratorSerializeTrait(tbblas::deeplearn::unit_type);

#endif /* TBBLAS_DEEPLEARN_UNITTYPE_H_ */
