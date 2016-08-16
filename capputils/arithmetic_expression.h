/*
 * arithmetic_expression.h
 *
 *  Created on: 2013-05-16
 *      Author: tombr
 */

#ifndef CAPPUTILS_ARITHMETIC_EXPRESSION_H_
#define CAPPUTILS_ARITHMETIC_EXPRESSION_H_

#include <capputils/capputils.h>

#include <string>

namespace capputils {

namespace util {

double eval_expression(const std::string& expressionString);

}

}


#endif /* ARITHMETIC_EXPRESSION_H_ */
