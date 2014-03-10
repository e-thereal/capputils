#pragma once
#ifndef CAPPUTILS_OUTPUTATTRIBUTE_H_
#define CAPPUTILS_OUTPUTATTRIBUTE_H_

#include <capputils/attributes/IAttribute.h>
#include <capputils/attributes/ShortNameAttribute.h>
#include <capputils/attributes/NoParameterAttribute.h>

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
