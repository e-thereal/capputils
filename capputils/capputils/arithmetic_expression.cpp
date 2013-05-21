/*
 * arithmetic_expression.cpp
 *
 *  Created on: 2013-05-16
 *      Author: tombr
 */

#include "arithmetic_expression.h"

#include "exprtk.hpp"

namespace capputils {

namespace util {

double eval_expression(const std::string& expressionString) {
  exprtk::symbol_table<double> symbol_table;
  symbol_table.add_constants();

  exprtk::expression<double> expression;
  expression.register_symbol_table(symbol_table);

  exprtk::parser<double> parser;
  parser.compile(expressionString, expression);

  return expression.value();
}

}

}
