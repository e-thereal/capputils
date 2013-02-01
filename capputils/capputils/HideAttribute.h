#pragma once
#ifndef CAPPUTILS_ATTRIBUTES_HIDEATTRIBUTE_H_
#define CAPPUTILS_ATTRIBUTES_HIDEATTRIBUTE_H_

#include <capputils/IAttribute.h>

namespace capputils {

namespace attributes {

class HideAttribute : public virtual capputils::attributes::IAttribute
{
public:
  HideAttribute(void);
};

capputils::attributes::AttributeWrapper* Hide();

}

}

#endif
