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
  std::string pattern; ///< Qt file dialog style (e.g. "Text (*.txt)")
  bool multipleSelection;

public:
  FilenameAttribute(const std::string& pattern = "All (*)", bool multipleSelection = false);
  virtual ~FilenameAttribute();

  virtual bool valid(const reflection::IClassProperty& property,
      const reflection::ReflectableClass& object);

  virtual std::string getLastMessage() const;

  bool getMultipleSelection() const;
  const std::string& getPattern() const;
  void setPattern(const std::string& pattern);
};

AttributeWrapper* Filename(const std::string& pattern = "All (*)",
    bool multipleSelection = false);

}

}

#endif /* FILENAMEATTRIBUTE_H_ */
