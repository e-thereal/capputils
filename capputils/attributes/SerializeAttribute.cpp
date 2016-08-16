#include <capputils/attributes/SerializeAttribute.h>

namespace capputils {

namespace attributes {

void serialize_trait<std::string>::writeToFile(const std::string& value, std::ostream& file)
{
  size_t count = value.size();
  file.write((char*)&count, sizeof(count));
  file.write((char*)value.c_str(), sizeof(std::string::value_type) * count);
}

void serialize_trait<std::string>::readFromFile(std::string& value, std::istream& file)
{
  size_t count = 0;
  file.read((char*)&count, sizeof(count));

  value.resize(count);
  file.read((char*)&value[0], sizeof(std::string::value_type) * count);
}

}

}
