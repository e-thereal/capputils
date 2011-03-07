#include "ScalarAttribute.h"

namespace capputils {

namespace attributes {

ScalarAttribute::ScalarAttribute() { }

ScalarAttribute::~ScalarAttribute() { }

AttributeWrapper* Scalar() {
  return new AttributeWrapper(new ScalarAttribute());
}

}

}