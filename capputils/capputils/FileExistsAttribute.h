/*
 * FileExists.h
 *
 *  Created on: Feb 11, 2011
 *      Author: tombr
 */

#ifndef FILEEXISTS_H_
#define FILEEXISTS_H_

#include "capputils.h"
#include "IAssertionAttribute.h"

namespace capputils {

namespace attributes {

class CAPPUTILS_API FileExistsAttribute: public virtual IAssertionAttribute {
private:
  std::string lastError;

public:
  FileExistsAttribute();
  virtual ~FileExistsAttribute();

  virtual bool valid(const reflection::IClassProperty& property,
      const reflection::ReflectableClass& object);

  virtual std::string getLastMessage() const;

  static bool exists(const std::string& filename);
};

AttributeWrapper* FileExists();

}

}

#endif /* FILEEXISTS_H_ */
