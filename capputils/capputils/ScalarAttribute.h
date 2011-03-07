#ifndef _CAPPUTILS_SCALARATTRIBUTE_H_
#define _CAPPUTILS_SCALARATTRIBUTE_H_

#include "IAttribute.h"

namespace capputils {

namespace attributes {

class ScalarAttribute : public virtual IAttribute {
public:
  ScalarAttribute();
  virtual ~ScalarAttribute();
};

AttributeWrapper* Scalar();

}

}

#endif