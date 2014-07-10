/*
 * ConvolutionType.h
 *
 *  Created on: Dec 9, 2013
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_CONVOLUTIONTYPE_H_
#define TBBLAS_DEEPLEARN_CONVOLUTIONTYPE_H_

#include <capputils/Enumerators.h>

namespace tbblas {

namespace deeplearn {

CapputilsEnumerator(convolution_type, Circular, Valid);

}

}

DefineEnumeratorSerializeTrait(tbblas::deeplearn::convolution_type);

#endif /* TBBLAS_DEEPLEARN_CONVOLUTIONTYPE_H_ */
