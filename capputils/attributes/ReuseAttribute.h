#pragma once

#ifndef _CAPPUTILS_REUSEATTRIBUTE_H_
#define _CAPPUTILS_REUSEATTRIBUTE_H_

#include <capputils/attributes/IAttribute.h>

namespace capputils {

namespace attributes {

class ReuseAttribute : public virtual IAttribute {
public:
  ReuseAttribute();
  virtual ~ReuseAttribute();
};

AttributeWrapper* Reuse();

}

}

#endif
