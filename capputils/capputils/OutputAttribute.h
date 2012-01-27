#pragma once
#ifndef _OUTPUTATTRIBUTE_H_
#define _OUTPUTATTRIBUTE_H_

#include "IAttribute.h"
#include "ShortNameAttribute.h"
#include "NoParameterAttribute.h"

namespace capputils {

namespace attributes {

class OutputAttribute : public virtual IAttribute, public NoParameterAttribute
{
public:
  OutputAttribute(void);
  virtual ~OutputAttribute(void);
};

class NamedOutputAttribute : public OutputAttribute, public ShortNameAttribute {
public:
  NamedOutputAttribute(const std::string& name);
};

AttributeWrapper* Output();
AttributeWrapper* Output(const std::string& name);

}

}

#endif
