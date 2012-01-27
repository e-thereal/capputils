#pragma once
#ifndef _INPUTATTRIBUTE_H_
#define _INPUTATTRIBUTE_H_

#include "IAttribute.h"
#include "ShortNameAttribute.h"
#include "NoParameterAttribute.h"

namespace capputils {

namespace attributes {

class InputAttribute : public virtual IAttribute, public NoParameterAttribute
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
