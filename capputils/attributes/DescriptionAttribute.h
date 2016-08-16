/**
 * \brief Contains the description attribute
 * \file DescriptionAttribute.h
 *
 * \date Feb 10, 2011
 * \author Tom Brosch
 */

#ifndef CAPPUTILS_DESCRIPTIONATTRIBUTE_H_
#define CAPPUTILS_DESCRIPTIONATTRIBUTE_H_

#include <capputils/capputils.h>
#include <capputils/attributes/IAttribute.h>

#include <string>

namespace capputils {

namespace attributes {

/**
 * \brief Attaches a description to a property.
 *
 * Use this attribute to document the intention of a property. The description
 * is presented to the user in several cases, e.g. when printing the usage information
 * and as a comment when writing a class as XML.
 */
class CAPPUTILS_API DescriptionAttribute: public virtual IAttribute {
private:
  std::string description;  ///< contains the description as a string

public:
  /**
   * \brief Constructs a new \c DescriptionAttribute
   *
   * \param[in] description The description as a string
   *
   * Use the Description() function which returns a wrapped DescriptionAttribute when passing
   * this attribute to an argument list.
   */
  DescriptionAttribute(const std::string& description);

  /**
   * \brief Destructs the attribute
   */
  virtual ~DescriptionAttribute();

  const std::string& getDescription() const;
};

/**
 * \brief Returns a wrapped DescriptionAttribute.
 */
CAPPUTILS_API AttributeWrapper* Description(const std::string& description);

}

}

#endif /* CAPPUTILS_DESCRIPTIONATTRIBUTE_H_ */
