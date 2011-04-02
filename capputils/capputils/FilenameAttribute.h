/*
 * FilenameAttribute.h
 *
 *  Created on: Mar 11, 2011
 *      Author: tombr
 */

#ifndef FILENAMEATTRIBUTE_H_
#define FILENAMEATTRIBUTE_H_

#include "IAssertionAttribute.h"

namespace capputils {

namespace attributes {

class FilenameAttribute: public virtual IAssertionAttribute {
private:
  std::string lastError;

public:
  FilenameAttribute();
  virtual ~FilenameAttribute();

  virtual bool valid(const reflection::IClassProperty& property,
      const reflection::ReflectableClass& object);

  virtual const std::string& getLastMessage() const;
};

AttributeWrapper* Filename();

}

}

#endif /* FILENAMEATTRIBUTE_H_ */