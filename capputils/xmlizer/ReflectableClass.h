/*
 * ReflectableClass.h
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#ifndef REFLECTABLECLASS_H_
#define REFLECTABLECLASS_H_

#include <sstream>
#include <string>
#include <vector>
#include <cstdarg>

#include "IAttribute.h"

namespace reflection {

class ReflectableClass;

template<class T>
T convertFromString(const std::string& value) {
  T result;
  std::stringstream s(value);
  s >> result;
  return result;
}

template<class T>
const std::string convertToString(const T& value) {
  std::stringstream s;
  s << value;
  return s.str();
}

/*** Template specializations for strings ***/

template<>
std::string convertFromString(const std::string& value);

class ClassProperty {
public:
  std::string name;
  std::vector<IAttribute*> attributes;

  const std::string (*getRawValue) (const ReflectableClass& object);
  void (*setValue) (ReflectableClass& object, const std::string& value);

  template<class T>
  T getValue(const ReflectableClass& object) const {
    return convertFromString<T>(getRawValue(object));
  }

  /* The last parameter is a list of IAttribute* which must be terminated
   * by null.
   */
  ClassProperty(const std::string& name,
      const std::string (*getValue) (const ReflectableClass& object),
      void (*setValue) (ReflectableClass& object, const std::string& value),
      ...)
      : name(name), getRawValue(getValue), setValue(setValue)
  {
    va_list args;
    va_start(args, setValue);

    for (IAttribute* attr = va_arg(args, AttributeWrapper).attribute; attr; attr = va_arg(args, AttributeWrapper).attribute)
      attributes.push_back(attr);
  }

  template<class AT>
  AT* getAttribute() {
    AT* attribute = 0;
    for (unsigned i = 0; i < attributes.size(); ++i) {
      IAttribute* att = attributes[i];
      attribute = dynamic_cast<AT*>(att);
      if (attribute != 0)
        return attribute;
    }
    return 0;
  }
};

class ReflectableClass {
public:
  virtual ~ReflectableClass();

public:
  /**
   * Remarks:
   * Properties are defined in a static vector. Therefore, the vector has to be part
   * of the specific class and can not be a field of the base class.
   */
  virtual std::vector<ClassProperty*>& getProperties() const = 0;
  virtual const std::string& getClassName() const = 0;

  ClassProperty* findProperty(const std::string& propertyName) const;

  bool hasProperty(const std::string& propertyName) const;

  void setProperty(const std::string& propertyName, const std::string& propertyValue);

  const std::string getProperty(const std::string& propertyName);
};

}

#define Property(name,type) \
private: type _##name; \
public: \
  type get##name() const { return _##name; } \
  void set##name(type value) { _##name = value; } \
protected: \
  static void set##name(::reflection::ReflectableClass& object, const std::string& value) { dynamic_cast<ClassType*>(&object)->set##name(::reflection::convertFromString<type>(value)); } \
  static const std::string get##name(const ::reflection::ReflectableClass& object) { return ::reflection::convertToString<type>(dynamic_cast<const ClassType*>(&object)->get##name()); }

#define InitReflectableClass(name)                    \
  typedef name ClassType;                         \
  protected: static std::vector< ::reflection::ClassProperty*> properties;                \
  public: virtual std::vector< ::reflection::ClassProperty*>& getProperties() const {  \
    initializeProperties();                         \
    return properties;                              \
  }                                               \
  public: static const std::string ClassName;     \
  public: virtual const std::string& getClassName() const { return ClassType::ClassName; } \
  private: static void initializeProperties();

#if defined(_MSC_VER)
#define DefineProperty(name, ...) \
  properties.push_back(new ::reflection::ClassProperty(#name, ClassType ::get##name, ClassType ::set##name, __VA_ARGS__, 0));
#else
#define DefineProperty(name, arguments...) \
  properties.push_back(new ::reflection::ClassProperty(#name, ClassType ::get##name, ClassType ::set##name, ##arguments, 0));
#endif

#define BeginPropertyDefinitions(name)   \
  std::vector< ::reflection::ClassProperty*> name :: properties;     \
  const std::string name :: ClassName = #name;  \
  void name :: initializeProperties() {   \
    static bool initialized = false;  \
    if (initialized) return;

#define EndPropertyDefinitions initialized = true; }

#endif /* XMLIZER_H_ */
