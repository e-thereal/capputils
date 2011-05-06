/*
 * ReflectableClass.h
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#ifndef REFLECTABLECLASS_H_
#define REFLECTABLECLASS_H_

#define CAPPUTILS_USE_CPP0x

#include <sstream>
#include <string>
#include <vector>
#include <cstdarg>
#include <ostream>
#include <istream>

#include "IAttribute.h"
#include "ClassProperty.h"
#include "AttributeExecuter.h"
#include "ReflectableAttribute.h"
#include "ReflectableClassFactory.h"

namespace capputils {

namespace reflection {

class ReflectableClass {
public:
  virtual ~ReflectableClass();

public:
  /**
   * Remarks:
   * Properties are defined in a static vector. Therefore, the vector has to be part
   * of the specific class and can not be a field of the base class.
   */
  virtual std::vector<IClassProperty*>& getProperties() const = 0;
  virtual std::vector< ::capputils::attributes::IAttribute*>& getAttributes() const = 0;
  virtual const std::string& getClassName() const = 0;
  virtual void toStream(std::ostream& stream) const;
  virtual void fromStream(std::istream& stream);

  IClassProperty* findProperty(const std::string& propertyName) const;

  bool hasProperty(const std::string& propertyName) const;

  void setProperty(const std::string& propertyName, const std::string& propertyValue);

  const std::string getProperty(const std::string& propertyName);

  template<class AT>
  AT* getAttribute() {
    AT* attribute = 0;
    const std::vector<attributes::IAttribute*>& attributes = getAttributes();
    for (unsigned i = 0; i < attributes.size(); ++i) {
      attribute = dynamic_cast<AT*>(attributes[i]);
      if (attribute != 0)
        return attribute;
    }
    return 0;
  }

protected:
  void addAttributes(std::vector< ::capputils::attributes::IAttribute*>* attributes, ...) const;
};

}

}

std::ostream& operator<< (std::ostream& stream, const capputils::reflection::ReflectableClass& object);
std::istream& operator>> (std::istream& stream, capputils::reflection::ReflectableClass& object);

#if !defined(CAPPUTILS_USE_CPP0x)
#define Property(name,type) \
private: type _##name; \
public: \
  type get##name() const { return _##name; } \
  void set##name(const type& value) { static ::capputils::reflection::IClassProperty* property = findProperty(#name); _##name = value; ::capputils::attributes::AttributeExecuter::Execute(*this, *property); } \
protected: \
  static void set##name(::capputils::reflection::ReflectableClass& object, const type& value) { dynamic_cast<ClassType*>(&object)->set##name(value); } \
  static type get##name(const ::capputils::reflection::ReflectableClass& object) { return dynamic_cast<const ClassType*>(&object)->get##name(); } \
  typedef type name##Type;
#define VirtualProperty(name,type) \
public: \
  type get##name() const; \
protected: \
  static void set##name(::capputils::reflection::ReflectableClass& object, type value) { } \
  static type get##name(const ::capputils::reflection::ReflectableClass& object) { return dynamic_cast<const ClassType*>(&object)->get##name(); } \
  typedef type name##Type;
#else
#define Property(name,type) \
private: type _##name; \
public: \
  type get##name() const { return _##name; } \
  void set##name(type value) { static ::capputils::reflection::IClassProperty* property = findProperty(#name); ::capputils::attributes::AttributeExecuter::ExecuteBefore(*this, *property); _##name = value; ::capputils::attributes::AttributeExecuter::ExecuteAfter(*this, *property); } \
protected: \
  static void set##name(::capputils::reflection::ReflectableClass& object, type value) { dynamic_cast<ClassType*>(&object)->set##name(value); } \
  static type get##name(const ::capputils::reflection::ReflectableClass& object) { return dynamic_cast<const ClassType*>(&object)->get##name(); }
#define VirtualProperty(name,type) \
public: \
  type get##name() const; \
protected: \
  static void set##name(::capputils::reflection::ReflectableClass& object, type value) { } \
  static type get##name(const ::capputils::reflection::ReflectableClass& object) { return dynamic_cast<const ClassType*>(&object)->get##name(); }
#endif

#define InitReflectableClass(name)                    \
  typedef name ClassType;                         \
  protected: static std::vector< ::capputils::reflection::IClassProperty*> properties;                \
  protected: static std::vector< ::capputils::attributes::IAttribute*> attributes;                \
  public: virtual std::vector< ::capputils::reflection::IClassProperty*>& getProperties() const {  \
    initializeProperties();                         \
    return properties;                              \
  }                                               \
  public: virtual std::vector< ::capputils::attributes::IAttribute*>& getAttributes() const {  \
    initializeAttributes();                        \
    return attributes;                             \
  }                                               \
  public: static const std::string ClassName;     \
  public: virtual const std::string& getClassName() const { return ClassType::ClassName; } \
  private: void initializeProperties() const; \
  private: void initializeAttributes() const; \
  private: static capputils::reflection::RegisterClass _registration; \
  public: static ReflectableClass* newInstance() { return new name(); } \
  public: static void deleteInstance(ReflectableClass* instance) { delete instance; }

#define InitAbstractReflectableClass(name)                    \
  typedef name ClassType;                         \
  protected: static std::vector< ::capputils::reflection::IClassProperty*> properties;                \
  protected: static std::vector< ::capputils::attributes::IAttribute*> attributes;                \
  public: virtual std::vector< ::capputils::reflection::IClassProperty*>& getProperties() const {  \
    initializeProperties();                         \
    return properties;                              \
  }                                               \
  public: virtual std::vector< ::capputils::attributes::IAttribute*>& getAttributes() const {  \
    initializeAttributes();                        \
    return attributes;                             \
  }                                               \
  public: static const std::string ClassName;     \
  public: virtual const std::string& getClassName() const { return ClassType::ClassName; } \
  private: void initializeProperties() const; \
  private: void initializeAttributes() const;

#if !defined(CAPPUTILS_USE_CPP0x)
#if defined(_MSC_VER)
#define DefineProperty(name, ...) \
  properties.push_back(new ::capputils::reflection::ClassProperty<name##Type>(#name, ClassType ::get##name, ClassType ::set##name, __VA_ARGS__, 0));
#define ReflectableProperty(name, ...) \
  properties.push_back(new ::capputils::reflection::ClassProperty<name##Type>(#name, ClassType ::get##name, ClassType ::set##name, __VA_ARGS__, capputils::attributes::Reflectable<name##Type>(), 0));
#else
#define DefineProperty(name, arguments...) \
  properties.push_back(new ::capputils::reflection::ClassProperty<name##Type>(#name, ClassType ::get##name, ClassType ::set##name, ##arguments, 0));
#define ReflectableProperty(name, arguments...) \
  properties.push_back(new ::capputils::reflection::ClassProperty<name##Type>(#name, ClassType ::get##name, ClassType ::set##name, ##arguments, capputils::attributes::Reflectable<name##Type>(), 0));
#endif
#else
#if defined(_MSC_VER)
#define DefineProperty(name, ...) \
  properties.push_back(new ::capputils::reflection::ClassProperty<decltype(((ClassType*)0)->get##name())>(#name, ClassType ::get##name, ClassType ::set##name, __VA_ARGS__, 0));
#define ReflectableProperty(name, ...) \
  properties.push_back(new ::capputils::reflection::ClassProperty<decltype(((ClassType*)0)->get##name())>(#name, ClassType ::get##name, ClassType ::set##name, __VA_ARGS__, capputils::attributes::Reflectable<decltype(((ClassType*)0)->get##name())>(), 0));
#else
#define DefineProperty(name, arguments...) \
  properties.push_back(new ::capputils::reflection::ClassProperty<decltype(((ClassType*)0)->get##name())>(#name, ClassType ::get##name, ClassType ::set##name, ##arguments, 0));
#define ReflectableProperty(name, arguments...) \
  properties.push_back(new ::capputils::reflection::ClassProperty<decltype(((ClassType*)0)->get##name())>(#name, ClassType ::get##name, ClassType ::set##name, ##arguments, capputils::attributes::Reflectable<decltype(((ClassType*)0)->get##name())>(), 0));
#endif
#endif

#define PROPERTY_ID properties.size()
#define ReflectableBase(BaseClass) \
  { \
  std::vector<capputils::reflection::IClassProperty*>& baseProperties = BaseClass::getProperties(); \
    for (unsigned i = 0; i < baseProperties.size(); ++i) \
      properties.push_back(baseProperties[i]); \
  }

#if defined(_MSC_VER)
#define BeginPropertyDefinitions(name, ...)   \
  std::vector< ::capputils::reflection::IClassProperty*> name :: properties;     \
  std::vector< ::capputils::attributes::IAttribute*> name :: attributes;     \
  const std::string name :: ClassName = #name;  \
  capputils::reflection::RegisterConstructor name::_constructor(#name, name::newInstance); \
  void name::initializeAttributes() const { \
    static bool initialized = false; \
    if (initialized) return; \
    addAttributes(&name::attributes, __VA_ARGS__, 0); \
    initialized = true; \
  } \
  void name :: initializeProperties() const {   \
    static bool initialized = false;  \
    if (initialized) return;
#define BeginAbstractPropertyDefinitions(name, ...)   \
  std::vector< ::capputils::reflection::IClassProperty*> name :: properties;     \
  std::vector< ::capputils::attributes::IAttribute*> name :: attributes;     \
  const std::string name :: ClassName = #name;  \
  void name::initializeAttributes() const { \
    static bool initialized = false; \
    if (initialized) return; \
    addAttributes(&name::attributes, __VA_ARGS__, 0); \
    initialized = true; \
  } \
  void name :: initializeProperties() const {   \
    static bool initialized = false;  \
    if (initialized) return;
#else
#define BeginPropertyDefinitions(name, args...)   \
  std::vector< ::capputils::reflection::IClassProperty*> name :: properties;     \
  std::vector< ::capputils::attributes::IAttribute*> name :: attributes;     \
  const std::string name :: ClassName = #name;  \
  capputils::reflection::RegisterClass name::_registration(#name, name::newInstance, name::deleteInstance); \
  void name::initializeAttributes() const { \
    static bool initialized = false; \
    if (initialized) return; \
    addAttributes(&name::attributes, ##args, 0); \
    initialized = true; \
  } \
  void name :: initializeProperties() const {   \
    static bool initialized = false;  \
    if (initialized) return;
#define BeginAbstractPropertyDefinitions(name, args...)   \
  std::vector< ::capputils::reflection::IClassProperty*> name :: properties;     \
  std::vector< ::capputils::attributes::IAttribute*> name :: attributes;     \
  const std::string name :: ClassName = #name;  \
  void name::initializeAttributes() const { \
    static bool initialized = false; \
    if (initialized) return; \
    addAttributes(&name::attributes, ##args, 0); \
    initialized = true; \
  } \
  void name :: initializeProperties() const {   \
    static bool initialized = false;  \
    if (initialized) return;
#endif
#define EndPropertyDefinitions initialized = true; }

#endif /* XMLIZER_H_ */
