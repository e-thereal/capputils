#include <capputils/attributes/ReuseAttribute.h>

namespace capputils {

namespace attributes {

ReuseAttribute::ReuseAttribute() { }
ReuseAttribute::~ReuseAttribute() { }

AttributeWrapper* Reuse() {
  return new AttributeWrapper(new ReuseAttribute());
}

}

}
