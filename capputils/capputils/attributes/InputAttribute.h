#pragma once
#ifndef CAPPUTILS_INPUTATTRIBUTE_H_
#define CAPPUTILS__INPUTATTRIBUTE_H_

#include <capputils/attributes/IAttribute.h>
#include <capputils/attributes/ShortNameAttribute.h>

namespace capputils {

namespace attributes {

class InputAttribute : public virtual IAttribute
{
public:
  InputAttribute(void);
  virtual ~InputAttribute(void);
};

class NamedInputAttribute : public InputAttribute, public ShortNameAttribute {
public:
  NamedInputAttribute(const std::string& name);
};

AttributeWrapper* Input();
AttributeWrapper* Input(const std::string& name);

}

}

#endif
