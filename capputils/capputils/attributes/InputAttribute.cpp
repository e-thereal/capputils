#include <capputils/attributes/InputAttribute.h>

namespace capputils {

namespace attributes {

InputAttribute::InputAttribute(void)
{
}


InputAttribute::~InputAttribute(void)
{
}

NamedInputAttribute::NamedInputAttribute(const std::string& name)
 : ShortNameAttribute(name) { }

AttributeWrapper* Input() {
  return new AttributeWrapper(new InputAttribute());
}

AttributeWrapper* Input(const std::string& name) {
  return new AttributeWrapper(new NamedInputAttribute(name));
}

}

}
