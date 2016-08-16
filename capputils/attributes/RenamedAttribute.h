/*
 * RenamedAttribute.h
 *
 *  Created on: Aug 26, 2014
 *      Author: tombr
 */

#ifndef CAPPTUTILS_RENAMEDATTRIBUTE_H_
#define CAPPTUTILS_RENAMEDATTRIBUTE_H_

#include <capputils/capputils.h>
#include <capputils/attributes/IAttribute.h>

#include <string>

namespace capputils {

namespace attributes {

class RenamedAttribute : public virtual IAttribute {
private:
  std::string newName;

public:
  RenamedAttribute(const std::string& newName);
  virtual ~RenamedAttribute();

  const std::string& getNewName() const;
};

AttributeWrapper* Renamed(const std::string& newName);

} /* namespace attributes */

} /* namespace capputils */

#endif /* CAPPTUTILS_RENAMEDATTRIBUTE_H_ */
