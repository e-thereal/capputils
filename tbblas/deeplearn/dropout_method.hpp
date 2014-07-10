/*
 * DropoutMethod.h
 *
 *  Created on: Apr 19, 2013
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_DROPOUTMETHOD_H_
#define TBBLAS_DEEPLEARN_DROPOUTMETHOD_H_

#include <capputils/Enumerators.h>

namespace tbblas {

namespace deeplearn {

CapputilsEnumerator(dropout_method, NoDrop, DropColumn, DropUnit);

}

}

DefineEnumeratorSerializeTrait(tbblas::deeplearn::dropout_method);

#endif /* TBBLAS_DEEPLEARN_DROPOUTMETHOD_H_ */
