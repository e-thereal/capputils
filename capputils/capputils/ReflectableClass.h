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

#include "IAttribute.h"
#include "ClassProperty.h"

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
  virtual const std::string& getClassName() const = 0;

  IClassProperty* findProperty(const std::string& propertyName) const;

  bool hasProperty(const std::string& propertyName) const;

  void setProperty(const std::string& propertyName, const std::string& propertyValue);

  const std::string getProperty(const std::string& propertyName);
};

}

}

#if !defined(CAPPUTILS_USE_CPP0x)
#define Property(name,type) \
private: type _##name; \
public: \
  type get##name() const { return _##name; } \
  void set##name(const type& value) { _##name = value; } \
protected: \
  static void set##name(::capputils::reflection::ReflectableClass& object, const type& value) { dynamic_cast<ClassType*>(&object)->set##name(value); } \
  static type get##name(const ::capputils::reflection::ReflectableClass& object) { return dynamic_cast<const ClassType*>(&object)->get##name(); } \
  typedef type name##Type;
#define VirtualProperty(name,type) \
public: \
  type get##name() const; \
protected: \
  static void set##name(::capputils::reflection::ReflectableClass& object, const type& value) { } \
  static type get##name(const ::capputils::reflection::ReflectableClass& object) { return dynamic_cast<const ClassType*>(&object)->get##name(); } \
  typedef type name##Type;
#else
#define Property(name,type) \
private: type _##name; \
public: \
  type get##name() const { return _##name; } \
  void set##name(const type& value) { _##name = value; } \
protected: \
  static void set##name(::capputils::reflection::ReflectableClass& object, const type& value) { dynamic_cast<ClassType*>(&object)->set##name(value); } \
  static type get##name(const ::capputils::reflection::ReflectableClass& object) { return dynamic_cast<const ClassType*>(&object)->get##name(); }
#define VirtualProperty(name,type) \
public: \
  type get##name() const; \
protected: \
  static void set##name(::capputils::reflection::ReflectableClass& object, const type& value) { } \
  static type get##name(const ::capputils::reflection::ReflectableClass& object) { return dynamic_cast<const ClassType*>(&object)->get##name(); }
#endif

#define InitReflectableClass(name)                    \
  typedef name ClassType;                         \
  protected: static std::vector< ::capputils::reflection::IClassProperty*> properties;                \
  public: virtual std::vector< ::capputils::reflection::IClassProperty*>& getProperties() const {  \
    initializeProperties();                         \
    return properties;                              \
  }                                               \
  public: static const std::string ClassName;     \
  public: virtual const std::string& getClassName() const { return ClassType::ClassName; } \
  private: static void initializeProperties();

#if !defined(CAPPUTILS_USE_CPP0x)
#if defined(_MSC_VER)
#define DefineProperty(name, ...) \
  properties.push_back(new ::capputils::reflection::ClassProperty<name##Type>(#name, ClassType ::get##name, ClassType ::set##name, __VA_ARGS__, 0));
#else
#define DefineProperty(name, arguments...) \
  properties.push_back(new ::capputils::reflection::ClassProperty<name##Type>(#name, ClassType ::get##name, ClassType ::set##name, ##arguments, 0));
#endif
#else
#if defined(_MSC_VER)
#define DefineProperty(name, ...) \
  properties.push_back(new ::capputils::reflection::ClassProperty<decltype(((ClassType*)0)->get##name())>(#name, ClassType ::get##name, ClassType ::set##name, __VA_ARGS__, 0));
#else
#define DefineProperty(name, arguments...) \
  properties.push_back(new ::capputils::reflection::ClassProperty<decltype(((ClassType*)0)->get##name())>(#name, ClassType ::get##name, ClassType ::set##name, ##arguments, 0));
#endif
#endif


#define BeginPropertyDefinitions(name)   \
  std::vector< ::capputils::reflection::IClassProperty*> name :: properties;     \
  const std::string name :: ClassName = #name;  \
  void name :: initializeProperties() {   \
    static bool initialized = false;  \
    if (initialized) return;

#define EndPropertyDefinitions initialized = true; }

#endif /* XMLIZER_H_ */
