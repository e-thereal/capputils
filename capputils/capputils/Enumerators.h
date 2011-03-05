/*
 * Enumerators.h
 *
 *  Created on: Mar 4, 2011
 *      Author: tombr
 */

#ifndef ENUMERATORS_H_
#define ENUMERATORS_H_


#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>

#include <vector>

#define DeclareEnum(name, values...) \
enum name { values }; \
namespace capputils { \
namespace reflection { \
template<> \
const std::string convertToString(const name& value); \
template<> \
name convertFromString(const std::string& value); \
} \
}

#define DefineEnum(name, values...) \
namespace capputils { \
namespace reflection { \
template<> \
const std::string convertToString(const name& value) {       \
  name valuesArr[] = {values};                   \
  std::string valuesString = #values;                       \
  std::vector<std::string> stringValues;                               \
  boost::char_separator<char> sep(", ");                         \
  boost::tokenizer<boost::char_separator<char> > tokens(valuesString, sep);     \
  BOOST_FOREACH(std::string t, tokens) {                       \
    stringValues.push_back(t);                               \
  }                                                       \
  for (unsigned i = 0; i < stringValues.size(); ++i)         \
    if (valuesArr[i] == value)                               \
      return stringValues[i];                                \
  return "unknown";                                       \
} \
template<> \
name convertFromString(const std::string& value) {       \
  name valuesArr[] = {values};                   \
  std::string valuesString = #values;                       \
  std::vector<std::string> stringValues;                               \
  boost::char_separator<char> sep(", ");                         \
  boost::tokenizer<boost::char_separator<char> > tokens(valuesString, sep);     \
  BOOST_FOREACH(std::string t, tokens) {                       \
    stringValues.push_back(t);                               \
  }                                                       \
  for (unsigned i = 0; i < stringValues.size(); ++i)         \
    if (stringValues[i].compare(value) == 0)                               \
      return valuesArr[i];                                \
  return valuesArr[0];                                       \
} \
} \
}

#endif /* ENUMERATORS_H_ */
