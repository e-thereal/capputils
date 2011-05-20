/*
 * ShortNameAttribute.h
 *
 *  Created on: May 6, 2011
 *      Author: tombr
 */

#ifndef SHORTNAMEATTRIBUTE_H_
#define SHORTNAMEATTRIBUTE_H_

#include "IAttribute.h"

#include <string>

namespace capputils {

namespace attributes {

class ShortNameAttribute : public virtual IAttribute {
private:
  std::string name;
public:
  ShortNameAttribute(const std::string& name);
  virtual ~ShortNameAttribute();

  const std::string& getName() const;
};

AttributeWrapper* ShortName(const std::string& name);

}

}

#endif /* SHORTNAMEATTRIBUTE_H_ */
