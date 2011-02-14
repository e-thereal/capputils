/*
 * FileExists.h
 *
 *  Created on: Feb 11, 2011
 *      Author: tombr
 */

#ifndef FILEEXISTS_H_
#define FILEEXISTS_H_

#include "IAssertionAttribute.h"

namespace xmlizer {

namespace attributes {

class FileExistsAttribute: public virtual IAssertionAttribute {
private:
  std::string lastError;

public:
  FileExistsAttribute();
  virtual ~FileExistsAttribute();

  virtual bool valid(const reflection::ClassProperty& property,
      const reflection::ReflectableClass& object);

  virtual const std::string& getLastMessage() const;
};

reflection::AttributeWrapper FileExists();

}

}

#endif /* FILEEXISTS_H_ */
