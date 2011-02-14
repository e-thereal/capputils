/*
 * DescriptionAttribute.h
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#ifndef DESCRIPTIONATTRIBUTE_H_
#define DESCRIPTIONATTRIBUTE_H_

#include <string>

#include "IAttribute.h"

namespace xmlizer {

namespace attributes {

class DescriptionAttribute: public virtual reflection::IAttribute {
private:
  std::string description;

public:
  DescriptionAttribute(const std::string& description);
  virtual ~DescriptionAttribute();

  const std::string& getDescription() const;
};

reflection::AttributeWrapper Description(const std::string& description);

}

}

#endif /* DESCRIPTIONATTRIBUTE_H_ */
