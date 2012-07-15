#ifndef CAPPUTILS_ATTRIBUTES_ENUMERATORATTRIBUTE_H_
#define CAPPUTILS_ATTRIBUTES_ENUMERATORATTRIBUTE_H_

#include "IAttribute.h"
#include "Enumerator.h"
#include "ReflectableClass.h"

#include <memory>

namespace capputils {

namespace attributes {

class IEnumeratorAttribute : public virtual IAttribute {
public:
  virtual std::shared_ptr<capputils::Enumerator> getEnumerator(const reflection::ReflectableClass& object,
      const reflection::IClassProperty* property) const = 0;
};

template<class T>
class EnumeratorAttribute : public virtual IEnumeratorAttribute {
public:
  virtual std::shared_ptr<capputils::Enumerator> getEnumerator(const reflection::ReflectableClass& object,
      const reflection::IClassProperty* property) const
  {
    using namespace capputils::reflection;
    const ClassProperty<T>* typedProperty = dynamic_cast<const ClassProperty<T>* >(property);
    if (typedProperty) {
      return std::shared_ptr<capputils::Enumerator>(new T(typedProperty->getValue(object)));
    } else {
      return std::shared_ptr<capputils::Enumerator>(0);
    }
  }
};

template<class T>
class EnumeratorAttribute<std::shared_ptr<T> > : public virtual IEnumeratorAttribute {
public:
  virtual std::shared_ptr<capputils::Enumerator> getEnumerator(const reflection::ReflectableClass& object,
        const reflection::IClassProperty* property) const
  {
    using namespace capputils::reflection;
    const ClassProperty<std::shared_ptr<T> >* typedProperty = dynamic_cast<const ClassProperty<std::shared_ptr<T> >* >(property);
    if (typedProperty) {
      return dynamic_pointer_cast<capputils::Enumerator>(typedProperty->getValue(object));
    }
    return std::shared_ptr<capputils::Enumerator>(0);
  }
};

template<class T>
class EnumeratorAttribute<T*> : public virtual IEnumeratorAttribute {
public:
  virtual std::shared_ptr<capputils::Enumerator> getEnumerator(const reflection::ReflectableClass& object,
        const reflection::IClassProperty* property) const
    {
      using namespace capputils::reflection;
      const ClassProperty<T*>* typedProperty = dynamic_cast<const ClassProperty<T*>* >(property);
      if (typedProperty) {
        return std::shared_ptr<capputils::Enumerator>(new T(*typedProperty->getValue(object)));
      }
      return std::shared_ptr<capputils::Enumerator>(0);
    }
};

template<class T>
AttributeWrapper* Enumerator() {
  return new AttributeWrapper(new EnumeratorAttribute<T>());
}

}

}

#endif
