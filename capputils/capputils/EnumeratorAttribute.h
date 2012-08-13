#ifndef CAPPUTILS_ATTRIBUTES_ENUMERATORATTRIBUTE_H_
#define CAPPUTILS_ATTRIBUTES_ENUMERATORATTRIBUTE_H_

#include "IAttribute.h"
#include "Enumerator.h"
#include "ReflectableClass.h"

#include <boost/shared_ptr.hpp>

namespace capputils {

namespace attributes {

class IEnumeratorAttribute : public virtual IAttribute {
public:
  virtual boost::shared_ptr<capputils::Enumerator> getEnumerator(const reflection::ReflectableClass& object,
      const reflection::IClassProperty* property) const = 0;
};

template<class T>
class EnumeratorAttribute : public virtual IEnumeratorAttribute {
public:
  virtual boost::shared_ptr<capputils::Enumerator> getEnumerator(const reflection::ReflectableClass& object,
      const reflection::IClassProperty* property) const
  {
    using namespace capputils::reflection;
    const ClassProperty<T>* typedProperty = dynamic_cast<const ClassProperty<T>* >(property);
    if (typedProperty) {
      return boost::shared_ptr<capputils::Enumerator>(new T(typedProperty->getValue(object)));
    } else {
      return boost::shared_ptr<capputils::Enumerator>();
    }
  }
};

template<class T>
class EnumeratorAttribute<boost::shared_ptr<T> > : public virtual IEnumeratorAttribute {
public:
  virtual boost::shared_ptr<capputils::Enumerator> getEnumerator(const reflection::ReflectableClass& object,
        const reflection::IClassProperty* property) const
  {
    using namespace capputils::reflection;
    const ClassProperty<boost::shared_ptr<T> >* typedProperty = dynamic_cast<const ClassProperty<boost::shared_ptr<T> >* >(property);
    if (typedProperty) {
      return boost::dynamic_pointer_cast<capputils::Enumerator>(typedProperty->getValue(object));
    }
    return boost::shared_ptr<capputils::Enumerator>();
  }
};

template<class T>
class EnumeratorAttribute<T*> : public virtual IEnumeratorAttribute {
public:
  virtual boost::shared_ptr<capputils::Enumerator> getEnumerator(const reflection::ReflectableClass& object,
        const reflection::IClassProperty* property) const
    {
      using namespace capputils::reflection;
      const ClassProperty<T*>* typedProperty = dynamic_cast<const ClassProperty<T*>* >(property);
      if (typedProperty) {
        return boost::shared_ptr<capputils::Enumerator>(new T(*typedProperty->getValue(object)));
      }
      return boost::shared_ptr<capputils::Enumerator>();
    }
};

template<class T>
AttributeWrapper* Enumerator() {
  return new AttributeWrapper(new EnumeratorAttribute<T>());
}

}

}

#endif
