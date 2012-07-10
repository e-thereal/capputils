#pragma once
#ifndef _INPUTATTRIBUTE_H_
#define _INPUTATTRIBUTE_H_

#include "IAttribute.h"
#include "ShortNameAttribute.h"

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
