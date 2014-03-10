#include <capputils/attributes/VolatileAttribute.h>

namespace capputils {

namespace attributes {

AttributeWrapper* Volatile() {
  return new AttributeWrapper(new VolatileAttribute());
}

}

}
