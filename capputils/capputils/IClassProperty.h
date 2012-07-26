/**
 * \brief Contains the IClassProperty interface.
 * \file IClassProperty.h
 *
 * \date Jan 7, 2011
 * \author Tom Brosch
 */

#pragma once

#ifndef CAPPUTILS_ICLASSPROPERTY_H
#define CAPPUTILS_ICLASSPROPERTY_H

#include <string>
#include <vector>
#include <cstdarg>

#include "IAttribute.h"
#include "Variant.h"

#include <typeinfo>

namespace capputils {

namespace reflection {

class ReflectableClass;

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
 *
 * The interface serves as a common base class for all \c ClassProperty objects.
 */
class IClassProperty {
public:

  /**
   * \brief Virtual destructor of an \c IClassProperty
   */
  virtual ~IClassProperty() { }

  /**
   * \brief Returns all attributes of a property
   *
   * \return  Vector containing the attributes of a property.
   */
  virtual const std::vector<attributes::IAttribute*>& getAttributes() const = 0;

  /**
   * \brief Adds an attribute to the property.
   *
   * \param[in] attribute Pointer to an attribute.
   *
   * \remarks
   * - The pointer is merely added to the attribute vector. No copies are created.
   */
  virtual void addAttribute(attributes::IAttribute* attribute) = 0;

  /**
   * \brief Returns the property name.
   *
   * \return  The name of the property.
   */
  virtual const std::string& getName() const = 0;

  /**
   * \brief Returns the value of the property as a string
   *
   * \param[in] object  Instance of \c ReflectableClass whose property is queried.
   * \return            Property value as a string.
   *
   * \remarks
   * - This method uses a \c Converter to convert the value to a string.
   */
  virtual std::string getStringValue(const ReflectableClass& object) const = 0;

  /**
   * \brief Sets the value of a string
   *
   * \param[out]  object  Reference to the \c ReflectabelClass whose property should be set.
   * \param[in]   value   New value of the property as a string.
   *
   * \remarks
   * - This method uses a \c Converter to convert the string to a typed value.
   */
  virtual void setStringValue(ReflectableClass& object, const std::string& value) const = 0;

  /**
   * \brief Typed setter method. Reads a value from one property of one object and writes that value to another property of
   * a possibly different object.
   *
   * \param[out]  object        Object whose property will be set.
   * \param[in]   fromObject    Reference to a \c ReflectableClass whose property will be read.
   * \param[in]   fromProperty  Reference to the property that will be read.
   *
   * \remarks
   * - The value is only transfered from one property to another, if both properties have exactly the same type.
   *   No casts are performed. Not even implicit casts. Also no down or up casts through a class hierarchy.
   */
  virtual void setValue(ReflectableClass& object, const ReflectableClass& fromObject, const IClassProperty* fromProperty) = 0;

  /**
   * \brief Returns the type of the property value.
   * \returns The type of the property value as a \c type_info object.
   */
  virtual const std::type_info& getType() const = 0;
  
  /**
   * \brief Returns the value of the property as a capputils::IVariant
   * \returns Value of the property wrapped into a capputils::IVariant
   */
  virtual IVariant* toVariant(const ReflectableClass& object) const = 0;

  /**
   * \brief Sets the value of a property from a variant
   *
   * \param[in] value   Value as a capputils::IVariant
   * \param[in] object  Reference to the ReflectableClass object
   *
   * The value is only written if the types of the variant and the property are identical
   *
   * \returns True if the value could be read from the variant.
   */
  virtual bool fromVariant(const IVariant& value, ReflectableClass& object) const = 0;

  /**
   * \brief Returns the first attribute in the attribute list matching the given template type
   *
   * \return The first attribute of the attribute list that can be successfully casted to the
   * given template type, or null if no such attribute exists.
   *
   * This methods tries a \c dynamic_cast for every attribute. If a cast is successful, the attribute
   * is returned.
   *
   * This method can be used to test, if a property has an attribute of a given type and use
   * the returned attribute for further processing.
   */
  template<class AT>
  AT* getAttribute() const {
    AT* attribute = 0;
    const std::vector<attributes::IAttribute*>& attributes = getAttributes();
    for (unsigned i = 0; i < attributes.size(); ++i) {
      attribute = dynamic_cast<AT*>(attributes[i]);
      if (attribute != 0)
        return attribute;
    }
    return 0;
  }
};

}

}

#endif /* CAPPUTILS_ICLASSPROPERTY_H_ */
