/**
 * \brief Contains the \c ClassProperty class and several \c Converter templates
 * \file ClassProperty.h
 *
 * \date Jan 7, 2011
 * \author Tom Brosch
 */

#pragma once

#ifndef CAPPUTILS_CLASSPROPERTY_H_
#define CAPPUTILS_CLASSPROPERTY_H_

#include "IClassProperty.h"

#include <sstream>
#include <iostream>
#include <typeinfo>

#include <boost/shared_ptr.hpp>

#include "ReflectionException.h"

namespace capputils {

namespace reflection {

class ReflectableClass;

/**
 * \brief Basic template for a \c Converter.
 *
 * \c Converters are used to convert the value
 * of a property to a string and vice versa. The \c fromString() method need not to exists.
 * This is handled by the template value parameter \c fromMethod which defaults to \c true.
 */
template<class T, bool fromMethod = true>
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

/**
 * \brief Models the property concept of a class
 *
 * A property of a class is a field of a class and associated setter and getter methods.
 * All three components are part of the class. A class property object provides the mapping
 * between class name and according getter and setter methods. It therefore contains field
 * for the property name and pointers to the static variants of the getter and setter methods.
 * The pointers to the static getter and setter methods might be removed in the future and replaced
 * by pointers to member functions.
 *
 * In addition, a class property contains a set of attributes which are stored in a vector of
 * type IAttribute*.
 */
template<class T>
class ClassProperty : public virtual IClassProperty
{
private:
  std::string name;                                             ///< Name of the property
  std::vector<attributes::IAttribute*> attributes;              ///< Vector with property attributes

  T (*getValueFunc) (const ReflectableClass& object);           ///< Pointer to the static getter method.
  void (*setValueFunc) (ReflectableClass& object, T value);     ///< Pointer to the static setter method.

  //mutable T value;

public:
  /* The last parameter is a list of IAttribute* which must be terminated
   * by null.
   */
  /**
   * \brief Creates a new instance of \c ClassProperty
   *
   * \param[in] name        Name of the property
   * \param[in] getValue    Pointer to the static getter method.
   * \param[in] setValue    Pointer to the static setter method.
   * \param[in] attributes  Null terminated list of AttributeWrapper*.
   *
   * The list of attributes must be terminated by null. Since variable argument
   * lists are a plain C feature, type information is lost when reading the arguments.
   * Hence the type of the arguments must be known. That is the reason why every
   * attribute is wrapped in an instance of \c AttributeWrapper.
   */
  ClassProperty(const std::string& name,
      T (*getValue) (const ReflectableClass& object),
      void (*setValue) (ReflectableClass& object, T value),
      ...)
      : name(name), getValueFunc(getValue), setValueFunc(setValue)
  {
    va_list args;
    va_start(args, setValue);

    for (attributes::AttributeWrapper* attrWrap = va_arg(args, attributes::AttributeWrapper*); attrWrap; attrWrap = va_arg(args, attributes::AttributeWrapper*)) {
      attributes.push_back(attrWrap->attribute);
      delete attrWrap;
    }
  }

  virtual const std::vector<attributes::IAttribute*>& getAttributes() const { return attributes; }
  virtual const std::string& getName() const { return name; }
  virtual void addAttribute(attributes::IAttribute* attribute) {
    attributes.push_back(attribute);
  }

  virtual std::string getStringValue(const ReflectableClass& object) const {
    return Converter<T>::toString(getValue(object));
  }

  virtual void setStringValue(ReflectableClass& object, const std::string& value) const {
    setValue(object, Converter<T>::fromString(value));
  }

  virtual const std::type_info& getType() const {
    return typeid(T);
  }

  virtual T getValue(const ReflectableClass& object) const {
    return getValueFunc(object);
  }

  virtual void setValue(ReflectableClass& object, T value) const {
    setValueFunc(object, value);
  }

  virtual void setValue(ReflectableClass& object, const ReflectableClass& fromObject, const IClassProperty* fromProperty) {
    const ClassProperty<T>* typedProperty = dynamic_cast<const ClassProperty<T>*>(fromProperty);
    if (typedProperty)
      setValue(object, typedProperty->getValue(fromObject));
  }
};

template<class T>
class ClassProperty<boost::shared_ptr<T> > : public virtual IClassProperty
{
private:
  std::string name;
  std::vector<attributes::IAttribute*> attributes;

  boost::shared_ptr<T> (*getValueFunc) (const ReflectableClass& object);
  void (*setValueFunc) (ReflectableClass& object, boost::shared_ptr<T> value);

public:
  /* The last parameter is a list of IAttribute* which must be terminated
   * by null.
   */
  ClassProperty(const std::string& name,
      boost::shared_ptr<T> (*getValue) (const ReflectableClass& object),
      void (*setValue) (ReflectableClass& object, boost::shared_ptr<T> value),
      ...)
      : name(name), getValueFunc(getValue), setValueFunc(setValue)
  {
    va_list args;
    va_start(args, setValue);

    for (attributes::AttributeWrapper* attrWrap = va_arg(args, attributes::AttributeWrapper*); attrWrap; attrWrap = va_arg(args, attributes::AttributeWrapper*)) {
      attributes.push_back(attrWrap->attribute);
      delete attrWrap;
    }
  }

  virtual ~ClassProperty() {
    for (unsigned i = 0; i < attributes.size(); ++i)
      delete attributes[i];
  }

  virtual const std::vector<attributes::IAttribute*>& getAttributes() const { return attributes; }
  virtual const std::string& getName() const { return name; }
  virtual void addAttribute(attributes::IAttribute* attribute) {
    attributes.push_back(attribute);
  }

  virtual std::string getStringValue(const ReflectableClass& object) const {
    return Converter<boost::shared_ptr<T> >::toString(getValue(object));
  }

  virtual void setStringValue(ReflectableClass& /*object*/, const std::string&/* value*/) const {
    //setValueFunc(object, new T(convertFromString<T>(value)));
    // TODO: static assert that getting pointer values from a string is not supported
    throw capputils::exceptions::ReflectionException("setting smart pointer values from a string is not supported");
  }

  virtual const std::type_info& getType() const {
    return typeid(boost::shared_ptr<T>);
  }

  virtual boost::shared_ptr<T> getValue(const ReflectableClass& object) const {
    return getValueFunc(object);
  }

  virtual void setValue(ReflectableClass& object, boost::shared_ptr<T> value) const {
    setValueFunc(object, value);
  }

  virtual void setValue(ReflectableClass& object, const ReflectableClass& fromObject, const IClassProperty* fromProperty) {
    const ClassProperty<boost::shared_ptr<T> >* typedProperty = dynamic_cast<const ClassProperty<boost::shared_ptr<T> >*>(fromProperty);
    if (typedProperty) {
      setValue(object, typedProperty->getValue(fromObject));
    }
  }
};

template<class T>
class ClassProperty<T*> : public virtual IClassProperty
{
private:
  std::string name;
  std::vector<attributes::IAttribute*> attributes;

  T* (*getValueFunc) (const ReflectableClass& object);
  void (*setValueFunc) (ReflectableClass& object, T* value);

public:
  /* The last parameter is a list of IAttribute* which must be terminated
   * by null.
   */
  ClassProperty(const std::string& name,
      T* (*getValue) (const ReflectableClass& object),
      void (*setValue) (ReflectableClass& object, T* value),
      ...)
      : name(name), getValueFunc(getValue), setValueFunc(setValue)
  {
    va_list args;
    va_start(args, setValue);

    for (attributes::AttributeWrapper* attrWrap = va_arg(args, attributes::AttributeWrapper*); attrWrap; attrWrap = va_arg(args, attributes::AttributeWrapper*)) {
      attributes.push_back(attrWrap->attribute);
      delete attrWrap;
    }
  }

  virtual ~ClassProperty() {
    for (unsigned i = 0; i < attributes.size(); ++i)
      delete attributes[i];
  }

  virtual const std::vector<attributes::IAttribute*>& getAttributes() const { return attributes; }
  virtual const std::string& getName() const { return name; }
  virtual void addAttribute(attributes::IAttribute* attribute) {
    attributes.push_back(attribute);
  }

  virtual std::string getStringValue(const ReflectableClass& object) const {
    return Converter<T*>::toString(getValue(object));
  }

  virtual void setStringValue(ReflectableClass& object, const std::string&/* value*/) const {
    //setValueFunc(object, new T(convertFromString<T>(value)));
    // TODO: static assert that getting pointer values from a string is not supported
    throw capputils::exceptions::ReflectionException("setting pointer values from a string is not supported");
  }

  virtual const std::type_info& getType() const {
    return typeid(T*);
  }

  virtual T* getValue(const ReflectableClass& object) const {
    return getValueFunc(object);
  }

  virtual void setValue(ReflectableClass& object, T* value) const {
    setValueFunc(object, value);
  }

  virtual void setValue(ReflectableClass& object, const ReflectableClass& fromObject, const IClassProperty* fromProperty) {
    const ClassProperty<T*>* typedProperty = dynamic_cast<const ClassProperty<T*>*>(fromProperty);
    if (typedProperty) {
      setValue(object, typedProperty->getValue(fromObject));
    }
  }
};

}

}

#endif /* CAPPUTILS_CLASSPROPERTY_H_ */
