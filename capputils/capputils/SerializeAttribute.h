#ifndef _CAPPUTILS_ATTRIBUTES_SERIALIZEATTRIBUTE_H_
#define _CAPPUTILS_ATTRIBUTES_SERIALIZEATTRIBUTE_H_

#include "IAttribute.h"

#include <cstdio>

namespace capputils {

namespace attributes {

class ISerializeAttribute : public IAttribute {
public:
  virtual ~SerializeAttribute() { }
};

template<class T>
class SerializeAttribute : public virtual ISerializeAttribute {

bool writeToFile(capputils::ClassProperty<T>* property, capputils::ReflectableClass* object, FILE* file) {
  
  //fwrite(
}

};

}

}

#endif /* _CAPPUTILS_ATTRIBUTES_SERIALIZEATTRIBUTE_H_ */