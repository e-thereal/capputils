#include "HideAttribute.h"

namespace capputils {

namespace attributes {

HideAttribute::HideAttribute(void) { }

AttributeWrapper* Hide() {
  return new AttributeWrapper(new HideAttribute());
}

}

}
