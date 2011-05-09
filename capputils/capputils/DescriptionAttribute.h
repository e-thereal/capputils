/*
 * DescriptionAttribute.h
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#ifndef DESCRIPTIONATTRIBUTE_H_
#define DESCRIPTIONATTRIBUTE_H_

#include <string>

#include "capputils.h"
#include "IAttribute.h"

namespace capputils {

namespace attributes {

class CAPPUTILS_API DescriptionAttribute: public virtual IAttribute {
private:
  std::string description;

public:
  DescriptionAttribute(const std::string& description);
  virtual ~DescriptionAttribute();

  const std::string& getDescription() const;
};

CAPPUTILS_API AttributeWrapper* Description(const std::string& description);

}

}

#endif /* DESCRIPTIONATTRIBUTE_H_ */
