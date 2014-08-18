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

#include <capputils/reflection/IClassProperty.h>

#include <sstream>
#include <iostream>
#include <typeinfo>

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

#include <capputils/exceptions/ReflectionException.h>
#include <capputils/Converter.h>

namespace capputils {

namespace reflection {

class ReflectableClass;

template<class T>
class ClassProperty;

template<class T>
struct string_value_trait {
  typedef T value_t;

  static void setValue(const ClassProperty<value_t>* const property, ReflectableClass& object, const std::string& value) {
    property->setValue(object, Converter<T>::fromString(value));
  }

  static std::string getValue(const ClassProperty<value_t>* const property, const ReflectableClass& object) {
    return Converter<value_t>::toString(property->getValue(object));
  }
};

// Set value is not supported for pointer types 

template<class T>
struct string_value_trait<boost::shared_ptr<T> > {
  typedef boost::shared_ptr<T> value_t;

  static void setValue(const ClassProperty<value_t>* const /*property*/, ReflectableClass& /*object*/, const std::string& /*value*/) {
    throw capputils::exceptions::ReflectionException("setting shared pointer values from a string is not supported");
  }

  static std::string getValue(const ClassProperty<value_t>* const property, const ReflectableClass& object) {
    return Converter<value_t>::toString(property->getValue(object));
  }
};

template<class T>
struct string_value_trait<boost::weak_ptr<T> > {
  typedef boost::weak_ptr<T> value_t;

  static void setValue(const ClassProperty<value_t>* const /*property*/, ReflectableClass& /*object*/, const std::string& /*value*/) {
    throw capputils::exceptions::ReflectionException("setting weak pointer values from a string is not supported");
  }

  static std::string getValue(const ClassProperty<value_t>* const property, const ReflectableClass& object) {
    boost::shared_ptr<T> temp = property->getValue(object).lock();
    return Converter<T*, false>::toString(temp.get());
  }
};

template<class T>
struct string_value_trait<T*> {
  typedef T* value_t;

  static void setValue(const ClassProperty<value_t>* const /*property*/, ReflectableClass& /*object*/, const std::string& /*value*/) {
    throw capputils::exceptions::ReflectionException("setting pointer values from a string is not supported");
  }

  static std::string getValue(const ClassProperty<value_t>* const property, const ReflectableClass& object) {
    return Converter<value_t>::toString(property->getValue(object));
  }
};

template<class T>
struct compare_trait {
  typedef T value_t;

  static int compare(const value_t& x, const value_t& y) {
    if (x == y)
      return 0;
    else if (x < y)
      return -1;
    else
      return 1;
  }
};

template<class T>
struct compare_trait<boost::shared_ptr<T> > {
  typedef boost::shared_ptr<T> value_t;

  static int compare(const value_t& x, const value_t& y) {
    return compare_trait<T*>::compare(x.get(), y.get());
  }
};

template<class T>
struct compare_trait<boost::weak_ptr<T> > {
  typedef boost::weak_ptr<T> value_t;

  static int compare(const value_t& x, const value_t& y) {
    return compare_trait<boost::shared_ptr<T> >::compare(x.lock(), y.lock());
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
public:
  typedef T value_t;

private:
  std::string name;                                             ///< Name of the property
  value_t defaultValue;                                         ///< Default value of the property
  std::vector<attributes::IAttribute*> attributes;              ///< Vector with property attributes

  value_t (*getValueFunc) (const ReflectableClass& object);           ///< Pointer to the static getter method.
  void (*setValueFunc) (ReflectableClass& object, value_t value);     ///< Pointer to the static setter method.
  void (*resetValueFunc) (ReflectableClass& object, value_t value);     ///< Pointer to the static resetter method.

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
  ClassProperty(const std::string& name, const value_t& defaultValue,
      value_t (*getValue) (const ReflectableClass& object),
      void (*setValue) (ReflectableClass& object, value_t value),
      void (*resetValue) (ReflectableClass& object, value_t value),
      ...)
      : name(name), defaultValue(defaultValue), getValueFunc(getValue), setValueFunc(setValue), resetValueFunc(resetValue)
  {
    va_list args;
    va_start(args, resetValue);

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
    return string_value_trait<value_t>::getValue(this, object);
    //return Converter<value_t>::toString(getValue(object));
  }

  virtual void setStringValue(ReflectableClass& object, const std::string& value) const {
    // this->setValue(object, Converter<value_t>::fromString(value));
    string_value_trait<value_t>::setValue(this, object, value);
    // boost::shared_ptr case: throw capputils::exceptions::ReflectionException("setting smart pointer values from a string is not supported");
  }

  virtual const std::type_info& getType() const {
    return typeid(value_t);
  }

  virtual T getValue(const ReflectableClass& object) const {
    return getValueFunc(object);
  }

  virtual IVariant* toVariant(const ReflectableClass& object) const {
    return new Variant<value_t>(getValue(object));
  }

  virtual bool fromVariant(const IVariant& value, ReflectableClass& object) const {
    const Variant<value_t>* typedValue = dynamic_cast<const Variant<value_t>* >(&value);
    if (typedValue) {
      setValue(object, typedValue->getValue());
      return true;
    } else {
      return false;
    }
  }

  virtual void setValue(ReflectableClass& object, value_t value) const {
    setValueFunc(object, value);
  }

  virtual void setValue(ReflectableClass& object, const ReflectableClass& fromObject, const IClassProperty* fromProperty) {
    const ClassProperty<value_t>* typedProperty = dynamic_cast<const ClassProperty<value_t>*>(fromProperty);
    if (typedProperty)
      setValue(object, typedProperty->getValue(fromObject));
  }

  virtual int compare(ReflectableClass& object, const ReflectableClass& fromObject, const IClassProperty* fromProperty) {
    const ClassProperty<value_t>* typedProperty = dynamic_cast<const ClassProperty<value_t>*>(fromProperty);
    assert(typedProperty);

    return compare_trait<value_t>::compare(getValue(object), typedProperty->getValue(fromObject));
  }

  virtual void resetValue(ReflectableClass& object) {
    resetValueFunc(object, defaultValue);
  }
};

}

}

#endif /* CAPPUTILS_CLASSPROPERTY_H_ */
