/*
 * ParameterAttribute.h
 *
 *  Created on: Dec 18, 2013
 *      Author: tombr
 */

#ifndef CAPPUTILS_ATTRIBUTES_PARAMETERATTRIBUTE_H_
#define CAPPUTILS_ATTRIBUTES_PARAMETERATTRIBUTE_H_

#include <capputils/attributes/IAttribute.h>

#include <string>

namespace capputils {

namespace attributes {

class ParameterAttribute : public virtual IAttribute {

private:
  std::string longName, shortName;

public:
  ParameterAttribute(const std::string& longName, const std::string& shortName = "");

  std::string getLongName() const;
  std::string getShortName() const;
};

AttributeWrapper* Parameter(const std::string& longName, const std::string& shortName = "");

} /* namespace attributes */

} /* namespace capputils */

#endif /* CAPPUTILS_ATTRIBUTES_PARAMETERATTRIBUTE_H_ */
