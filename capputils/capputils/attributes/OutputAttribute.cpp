
#include <capputils/attributes/OutputAttribute.h>

namespace capputils {

namespace attributes {

OutputAttribute::OutputAttribute(void)
{
}


OutputAttribute::~OutputAttribute(void)
{
}

NamedOutputAttribute::NamedOutputAttribute(const std::string& name)
 : ShortNameAttribute(name)
{
}

AttributeWrapper* Output() {
  return new AttributeWrapper(new OutputAttribute());
}

AttributeWrapper* Output(const std::string& name) {
  return new AttributeWrapper(new NamedOutputAttribute(name));
}

}

}
