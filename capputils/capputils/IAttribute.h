/**
 * \brief Declares the base interface \c IAttribute of all attributes
 *  \file IAttribute.h
 *
 *  \date   Feb 10, 2011
 *  \author Tom Brosch
 */

#ifndef CAPPUTILS_IATTRIBUTE_H_
#define CAPPUTILS_IATTRIBUTE_H_

namespace capputils {

/** \brief Attributes are defined in its own namespace \c attributes */
namespace attributes {

/**
 * \brief This is the base interface of all attributes
 */
class IAttribute {
public:
  /**
   * \brief Virtual destructor
   */
  virtual ~IAttribute();
};

/**
 * \brief Attribute wrappers are used internally
 *
 * Some variadic macros and functions take a list of attributes as arguments. Since variadic
 * macros and functions are pure C features, some compilers discard type informations when
 * passing argument lists. In order to preserve the type information of an attribute, every
 * attribute is wrapped using an attribute wrapper. Most attributes contain a function which
 * returns an already wrapped attribute.
 */
class AttributeWrapper {
public:
  IAttribute* attribute;  ///< An internal pointer to the attribute. Type information is preserved

  /**
   * \brief Constructs a new attribute wrapper instance
   * \param[in] attribute Pointer to the attribute that needs to be wrapped.
   */
  AttributeWrapper(IAttribute* attribute) : attribute (attribute) { }
};

}

}

#endif /* CAPPUTILS_IATTRIBUTE_H_ */
