#ifndef _CAPPUTILS_REFLECTABLEATTRIBUTE_H_
#define _CAPPUTILS_REFLECTABLEATTRIBUTE_H_

#include <capputils/attributes/IReflectableAttribute.h>
#include <capputils/reflection/ClassProperty.h>

#include <boost/shared_ptr.hpp>

namespace capputils {

namespace attributes {

template<class T>
class ReflectableAttribute : public virtual IReflectableAttribute {
private:
  mutable T value;

public:
  virtual reflection::ReflectableClass* getValuePtr(const reflection::ReflectableClass& object,
      const reflection::IClassProperty* property) const
  {
    using namespace capputils::reflection;
    const ClassProperty<T>* typedProperty = dynamic_cast<const ClassProperty<T>* >(property);
    if (typedProperty) {
      value = typedProperty->getValue(object);
      return &value;
    }
    return 0;
  }

  virtual void setValuePtr(reflection::ReflectableClass& object,
      reflection::IClassProperty* property, reflection::ReflectableClass* valuePtr) const
  {
    using namespace capputils::reflection;
    ClassProperty<T>* typedProperty = dynamic_cast<ClassProperty<T>* >(property);
    if (typedProperty) {
      typedProperty->setValue(object, *((T*)valuePtr));
    }
  }

  virtual bool isPointer() const {
    return false;
  }

  virtual bool isSmartPointer() const {
    return false;
  }
};

template<class T>
class ReflectableAttribute<boost::shared_ptr<T> > : public virtual IReflectableAttribute {
public:
  virtual reflection::ReflectableClass* getValuePtr(const reflection::ReflectableClass& object,
        const reflection::IClassProperty* property) const
  {
    using namespace capputils::reflection;
    const ClassProperty<boost::shared_ptr<T> >* typedProperty = dynamic_cast<const ClassProperty<boost::shared_ptr<T> >* >(property);
    if (typedProperty) {
      return typedProperty->getValue(object).get();
    }
    return 0;
  }

  virtual void setValuePtr(reflection::ReflectableClass& object,
        reflection::IClassProperty* property, reflection::ReflectableClass* valuePtr) const
  {
    using namespace capputils::reflection;
    ClassProperty<boost::shared_ptr<T> >* typedProperty = dynamic_cast<ClassProperty<boost::shared_ptr<T> >* >(property);
    if (typedProperty) {
      boost::shared_ptr<T> smartPtr((T*)valuePtr);
      typedProperty->setValue(object, smartPtr);
    }
  }

  virtual bool isPointer() const {
    return false;
  }

  virtual bool isSmartPointer() const {
    return true;
  }
};

template<class T>
class ReflectableAttribute<T*> : public virtual IReflectableAttribute {
public:
  virtual reflection::ReflectableClass* getValuePtr(const reflection::ReflectableClass& object,
        const reflection::IClassProperty* property) const
    {
      using namespace capputils::reflection;
      const ClassProperty<T*>* typedProperty = dynamic_cast<const ClassProperty<T*>* >(property);
      if (typedProperty) {
        return typedProperty->getValue(object);
      }
      return 0;
    }

  virtual void setValuePtr(reflection::ReflectableClass& object,
        reflection::IClassProperty* property, reflection::ReflectableClass* valuePtr) const
  {
    using namespace capputils::reflection;
    ClassProperty<T*>* typedProperty = dynamic_cast<ClassProperty<T*>* >(property);
    if (typedProperty) {
      typedProperty->setValue(object, (T*)valuePtr);
    }
  }

  virtual bool isPointer() const {
    return true;
  }

  virtual bool isSmartPointer() const {
    return false;
  }
};

template<class T>
AttributeWrapper* Reflectable() {
  return new AttributeWrapper(new ReflectableAttribute<T>());
}

}

}

#endif
