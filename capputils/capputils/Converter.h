/*
 * Converter.h
 *
 *  Created on: Jul 25, 2012
 *      Author: tombr
 */

#ifndef CAPPUTILS_CONVERTER_H_
#define CAPPUTILS_CONVERTER_H_

#include <capputils/capputils.h>

#include <string>
#include <sstream>
#include <vector>

#include <boost/weak_ptr.hpp>
#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>

#include <capputils/arithmetic_expression.h>

namespace capputils {

namespace reflection {

/**
 * \brief Basic template for a \c Converter.
 *
 * \c Converters are used to convert the value
 * of a property to a string and vice versa. The \c fromString() method need not to exists.
 * This is handled by the template value parameter \c fromMethod which defaults to \c true.
 */
template<class T, bool fromMethod = true, class Enable = void>
class Converter {
public:
  /**
   * \brief Converts a string to a value of type \c T
   *
   * \param[in] value Value as a \c std::string
   * \returns   The value as an instance of type \c T.
   *
   * This method uses a \c std::stringstream for the conversion. This works for all
   * types which implement the \c >> operator.
   */
  static T fromString(const std::string& value) {
    T result;
    std::stringstream s(value);
    s >> result;
    return result;
  }

  /**
   * \brief Converts a value of type \c T to a \c std::string
   *
   * \param[in] value Value of type \c T
   * \return    The value as a \c std::string.
   *
   * This method uses a \c std::stringstream for the conversion. This works for all
   * types which implement the \c << operator.
   */
  static std::string toString(const T& value) {
    std::stringstream s;
    s << value;
    return s.str();
  }
};

/**
 * \brief Specialized \c Converter template without the \c fromString() method.
 *
 * Templates not featuring a \c fromString() method are used to convert pointers,
 * since it is not possible to create an instance of the correct type without further
 * type information. Furthermore, properties with a pointer type could be a pointer to
 * an abstract class.
 */
template<class T>
class Converter<T, false> {
public:
  /**
   * \brief Converts a value of type \c T to a \c std::string
   *
   * \param[in] value Value of type \c T
   * \return    The value as a \c std::string.
   *
   * This method uses a \c std::stringstream for the conversion. This works for all
   * types which implement the \c << operator.
   */
  static std::string toString(const T& value) {
    std::stringstream s;
    s << value;
    return s.str();
  }
};

/*** Template specializations for strings (from string variants) ***/

/*template<class T>
class Converter<T*> {
public:
  static std::string toString(const T* value) {
    if (value) {
      return Converter<T, false>::toString(*value);
    } else {
      return "<null>";
    }
  }
};*/

/**
 * \brief Generic converter to convert from and to a \c std::vector<T>
 */
template<class T>
class Converter<std::vector<T>, true> {
public:

  /**
   * \brief Converts from a \c std::string to a \c std::vector<T>.
   *
   * \param[in] value Values as a \c std::string.
   * \return    A \c std::vector containing the parsed values.
   *
   * A stringstream is used to divide the input string into substring. The \c Converter<T>
   * class is used to convert the substring to their values.
   */
  static std::vector<T> fromString(const std::string& value) {
    std::string result;
    std::stringstream s(value);
    std::vector<T> vec;
    if (value.size() == 0)
        return vec;
    while (!s.eof()) {
      s >> result;
      vec.push_back(Converter<T>::fromString(result));
    }
    return vec;
  }

  /**
   * \brief Converts a vector of values into a single string. Values are separated by spaces.
   *
   * \param[in] value Vector containing the values
   * \return    A string representation of the input values.
   */
  static std::string toString(const std::vector<T>& value) {
    std::stringstream s;
    if (value.size())
      s << value[0];
    for (unsigned i = 1; i < value.size(); ++i)
      s << " " << value[i];
    return s.str();
  }
};

/**
 * \brief Specialized template of the vector converter without a \c fromString() method.
 */
template<class T>
class Converter<std::vector<T>, false> {
public:
  /**
   * \brief Converts a vector of values into a single string. Values are separated by spaces.
   *
   * \param[in] value Vector containing the values
   * \return    A string representation of the input values.
   */
  static std::string toString(const std::vector<T>& value) {
    std::stringstream s;
    if (value.size())
      s << value[0];
    for (unsigned i = 1; i < value.size(); ++i)
      s << " " << value[i];
    return s.str();
  }
};

/**
 * \brief Specialized \c Converter template for strings.
 *
 * This is necessary in order to keep white spaces in strings.
 */
template<>
class Converter<std::string> {
public:
  /**
   * \brief Returns a copy of the string
   *
   * \param[in] value Value as a string
   * \return    Copy of \a value.
   */
  static std::string fromString(const std::string& value) {
    return std::string(value);
  }

  /**
   * \brief Returns a copy of the string
   *
   * \param[in] value Value as a string
   * \return    Copy of \a value.
   */
  static std::string toString(const std::string& value) {
    return std::string(value);
  }
};

/**
 * \brief Specialized converter to convert a vector of string
 *
 * Unlike the general vector converter, values are separated by white spaces and enclosed
 * in quotation marks. This allows for white spaces in strings.
 */
template<>
class Converter<std::vector<std::string>, true> {
public:
  static std::vector<std::string> fromString(const std::string& value) {
    std::vector<std::string> vec;
    std::string str;
    bool withinString = false;
    for (unsigned i = 0; i < value.size(); ++i) {
      if (withinString) {
        if (value[i] == '\"') {
          withinString = false;
          vec.push_back(str);
          str = "";
        } else {
          str += value[i];
        }
      } else {
        if (value[i] == '\"')
          withinString = true;
      }
    }

    return vec;
  }

  static std::string toString(const std::vector<std::string>& value) {
    std::stringstream s;
    if (value.size())
      s << "\"" << value[0] << "\"";
    for (unsigned i = 1; i < value.size(); ++i)
      s << " \"" << value[i] << "\"";
    return s.str();
  }
};

template<>
class Converter<std::vector<std::string>, false> {
public:
  static std::string toString(const std::vector<std::string>& value) {
    std::stringstream s;
    if (value.size())
      s << "\"" << value[0] << "\"";
    for (unsigned i = 1; i < value.size(); ++i)
      s << " \"" << value[i] << "\"";
    return s.str();
  }
};

template<class T>
class Converter<T, true, typename boost::enable_if<boost::is_arithmetic<T> >::type> {
public:
  static T fromString(const std::string& value) {
    return util::eval_expression(value);
  }

  static std::string toString(const T& value) {
    std::stringstream s;
    s << value;
    return s.str();
  }
};

}

}

#endif /* CAPPUTILS_CONVERTER_H_ */
