/*
 * Enumerators.h
 *
 *  Created on: Mar 4, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_ENUMERATORS_H_
#define CAPPUTILS_ENUMERATORS_H_

#include <capputils/AbstractEnumerator.h>
#include <capputils/TypeTraits.h>
#include <capputils/attributes/SerializeAttribute.h>

#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>

#include <istream>
#include <ostream>
#include <vector>

#define CapputilsEnumerator(name, ...) \
class name : public capputils::AbstractEnumerator { \
public: \
  enum enum_type {__VA_ARGS__}; \
\
public: \
  name () { \
    initialize(); \
    value = getValues()[0]; \
  } \
\
  name (const enum_type& value) { \
    initialize(); \
    this->value = getValues()[value]; \
  } \
\
  void initialize() { \
    static bool initialized = false; \
    if (initialized) \
      return; \
\
    std::string valuesString = #__VA_ARGS__; \
    boost::char_separator<char> sep(", "); \
    boost::tokenizer<boost::char_separator<char> > tokens(valuesString, sep); \
    std::vector<std::string>& values = getValues(); \
    BOOST_FOREACH(std::string t, tokens) { \
      values.push_back(t); \
    } \
\
    initialized = true; \
  } \
  \
  virtual std::vector<std::string>& getValues() const { \
    static std::vector<std::string> values; \
    return values; \
  } \
  operator std::string() const { \
    return value; \
  } \
  \
  virtual int toInt() const { \
    std::vector<std::string>& values = getValues(); \
    for (unsigned i = 0; i < values.size(); ++i) { \
      if (value.compare(values[i]) == 0) \
        return i; \
    } \
    return -1; \
  } \
  operator int() const { \
    return toInt(); \
  } \
  void operator=(const std::string& value) { \
    this->value = value; \
  }\
};

#define DefineEnumeratorSerializeTrait(name) \
namespace capputils { namespace attributes { \
template<> \
class serialize_trait<name> { \
public: \
  static void writeToFile(const name& value, std::ostream& file) { \
    int i = value; \
    serialize_trait<int>::writeToFile(i, file); \
  } \
  static void readFromFile(name& value, std::istream& file) { \
    int i = 0; \
    serialize_trait<int>::readFromFile(i, file); \
    value = (name::enum_type)i; \
  } \
}; \
}\
}

#define ReflectableEnum(...) CapputilsEnumerator(__VA_ARGS__)
#define DefineEnum(...)

#endif /* CAPPUTILS_ENUMERATORS_H_ */
