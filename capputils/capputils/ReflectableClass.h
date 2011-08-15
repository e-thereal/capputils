/**
 * \brief Contains the class declaration of ReflecableClass and some useful macros
 * \file ReflectableClass.h
 *
 *  \date   Jan 7, 2011
 *  \author Tom Brosch
 *  
 *  You must call InitReflectableClass and BeginProperyDefintions and EndPropertyDefinitions.
 *  See example for help. (minimal class)
 */

#ifndef CAPPUTILS_REFLECTABLECLASS_H_
#define CAPPUTILS_REFLECTABLECLASS_H_

#define CAPPUTILS_USE_CPP0x

#include <sstream>
#include <string>
#include <vector>
#include <cstdarg>
#include <ostream>
#include <istream>

#include "capputils.h"
#include "IAttribute.h"
#include "ClassProperty.h"
#include "AttributeExecuter.h"
#include "ReflectableAttribute.h"
#include "ReflectableClassFactory.h"

/** \brief main namespace */
namespace capputils {

/** \brief Everything that has to do with reflection comes in here */
namespace reflection {

/**
 * \brief Base class of all reflectable classes
 * 
 * \remarks
 * Properties are defined in a static vector. Therefore, the vector has to be part
 * of the specific class and can not be a field of the base class.
 */
class ReflectableClass {
public:
  virtual ~ReflectableClass();

public:
  /**
   * \brief Returns all properties of the class including properties of all of its base classes
   * 
   * \return Vector containing pointers to all defined class properties.
   * 
   * \remark
   * - This method is automatically defined by the BeginPropertyDefinitions macro.
   */
  virtual std::vector<IClassProperty*>& getProperties() const = 0;

  /**
   * \brief Returns all attributes of a class.
   * 
   * \return Vector containg pointers to all attributes of the class.
   * 
   * \remark
   * - This method is automatically defined by the BeginPropertyDefinitions macro.
   */
  virtual std::vector< ::capputils::attributes::IAttribute*>& getAttributes() const = 0;

  /**
   * \brief Returns the full class name including its namespaces
   * 
   * The returned class name is of the form namespace::classname
   * 
   * \return Class name including its namespaces
   * 
   * \remark
   * - This method is automatically defined by the BeginPropertyDefinitions macro.
   */
  virtual const std::string& getClassName() const = 0;

  /**
   * \brief Overload this method to provide a customized way of writing a class to a stream
   * 
   * \param[in, out] stream  Stream where the class will be written to
   */
  virtual void toStream(std::ostream& stream) const;

  /**
   * \brief Overload this method to provide a customized way of reading a class from a stream
   * 
   * \param[in, out] stream Stream from which the class properties will be read from
   */
  virtual void fromStream(std::istream& stream);

  /** 
   * \brief Returns the pointer to a class property given its name
   * 
   * \param[in] propertyName Name of the property according property.
   * 
   * \return Pointer to the class property or 0 if there is no property with the given name.
   */
  IClassProperty* findProperty(const std::string& propertyName) const;

  /**
   * \brief Returns the index of a property given its name
   *
   * \param[out]  index         Will be set to the index of the property if a property with the given name exists.
   * \param[in]   propertyName  Name of the property
   *
   * \return true iff a property with the given name could be found.
   */
  bool getPropertyIndex(unsigned& index, const std::string& propertyName) const;

  /**
   * \brief Returns true of the current class has a property of the given name
   * 
   * \param[in] propertyName Name of the according property.
   * 
   * \return True, iff the class has a property with the given name
   */
  bool hasProperty(const std::string& propertyName) const;

  /**
   * \brief Sets the value of a given property by the properties name using a string representing the value
   * 
   * \param[in] propertyName Name of the according property
   * \param[in] propertyValue New value of the property as a string
   * 
   * \remark
   * - This method is not type-safe. Use findProperty() and use one of the type-safe setters of IClassProperty
   *   to set the value safely.
   */
  void setProperty(const std::string& propertyName, const std::string& propertyValue);

  /**
   * \brief Returns the value of a property as a string
   * 
   * \param[in] propertyName Name of the according property
   * 
   * \return Value of the property as a string
   */
  const std::string getProperty(const std::string& propertyName);

  /**
   * Returns a pointer to an attribute given its type
   * 
   * \return A pointer to an attribute if the class contains an attribute of the given type, 0 otherwise.
   * 
   * The attribute type is given as the type parameter of the template method.
   * 
   * \remark
   * - This method can be used to check if a class is flagged with a particular attribute.
   */
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

/**
 * \brief Converts a type name string obtained by \c std::type_info::name() into a standardized format
 *
 * \param[in] typeName  Type name in a compiler dependent format.
 * \return              Type name in the format \c <namespace>::<classname>
 * \remarks
 * - The implementation depends on the compiler in use.
 * - Currently supported compilers are MSVC 9 and 10 and GCC 4.
 */
std::string trimTypeName(const char* typeName); 

}

}

std::ostream& operator<< (std::ostream& stream, const capputils::reflection::ReflectableClass& object);
std::istream& operator>> (std::istream& stream, capputils::reflection::ReflectableClass& object);

/**
 * \def Property(name, type)
 * \brief Declares a property
 * 
 * The first parameter is the name of the property. Capitalization is recommended. The second parameter is the type of the property.
 * The macro will define according getter and setter methods.
 * 
 * For example:
 * \code
 *  Property(Name, std::string)
 * \endcode
 * will generate
 * \code
 *   std::string getName() const { ... }
 *   void setName(std::string value) { ... }
 * \endcode
 *
 * \def InitReflectableClass(name)
 * \brief Must be called at the beginning of every reflectable class
 *
 * The parameter denotes the name of the class. This macro declares all static fields and methods
 * used to store and initialize class meta information like:
 * - Properties
 * - Attributes
 * - The class name
 */

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
    initializeClass();                            \
    return properties;                            \
  }                                               \
  public: virtual std::vector< ::capputils::attributes::IAttribute*>& getAttributes() const {  \
    initializeClass();                            \
    return attributes;                            \
  }                                               \
  public: static const std::string ClassName;     \
  public: virtual const std::string& getClassName() const { return ClassType::ClassName; } \
  private: void initializeClass() const; \
  private: static capputils::reflection::RegisterClass _registration; \
  public: static ReflectableClass* newInstance() { return new name(); } \
  public: static void deleteInstance(ReflectableClass* instance) { delete instance; }

#define InitAbstractReflectableClass(name)                    \
  typedef name ClassType;                         \
  protected: static std::vector< ::capputils::reflection::IClassProperty*> properties;                \
  protected: static std::vector< ::capputils::attributes::IAttribute*> attributes;                \
  public: virtual std::vector< ::capputils::reflection::IClassProperty*>& getProperties() const {  \
    initializeClass();                         \
    return properties;                              \
  }                                               \
  public: virtual std::vector< ::capputils::attributes::IAttribute*>& getAttributes() const {  \
    initializeClass();                        \
    return attributes;                             \
  }                                               \
  public: static const std::string ClassName;     \
  public: virtual const std::string& getClassName() const { return ClassType::ClassName; } \
  private: void initializeClass() const;

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
    std::vector<capputils::attributes::IAttribute*>& baseAttributes = BaseClass::getAttributes(); \
    for (unsigned i = 0; i < baseAttributes.size(); ++i) \
      attributes.push_back(baseAttributes[i]); \
  }

#if defined(_MSC_VER)
#define BeginPropertyDefinitions(cname, ...)   \
  std::vector< ::capputils::reflection::IClassProperty*> cname :: properties;     \
  std::vector< ::capputils::attributes::IAttribute*> cname :: attributes;     \
  const std::string cname :: ClassName = capputils::reflection::trimTypeName(typeid(cname).name());  \
  capputils::reflection::RegisterClass cname::_registration(capputils::reflection::trimTypeName(typeid(cname).name()), cname::newInstance, cname::deleteInstance); \
  void cname::initializeClass() const { \
    static bool initialized = false; \
    if (initialized) return; \
    addAttributes(&cname::attributes, __VA_ARGS__, 0);

#define BeginAbstractPropertyDefinitions(cname, ...)   \
  std::vector< ::capputils::reflection::IClassProperty*> cname :: properties;     \
  std::vector< ::capputils::attributes::IAttribute*> cname :: attributes;     \
  const std::string cname :: ClassName = capputils::reflection::trimTypeName(typeid(cname).name());  \
  void cname::initializeClass() const { \
    static bool initialized = false; \
    if (initialized) return; \
    addAttributes(&cname::attributes, __VA_ARGS__, 0);
#else
#define BeginPropertyDefinitions(cname, args...)   \
  std::vector< ::capputils::reflection::IClassProperty*> cname :: properties;     \
  std::vector< ::capputils::attributes::IAttribute*> cname :: attributes;     \
  const std::string cname :: ClassName = capputils::reflection::trimTypeName(typeid(cname).name());  \
  capputils::reflection::RegisterClass cname::_registration(capputils::reflection::trimTypeName(typeid(cname).name()), cname::newInstance, cname::deleteInstance); \
  void cname::initializeClass() const { \
    static bool initialized = false; \
    if (initialized) return; \
    addAttributes(&cname::attributes, ##args, 0);

#define BeginAbstractPropertyDefinitions(cname, args...)   \
  std::vector< ::capputils::reflection::IClassProperty*> cname :: properties;     \
  std::vector< ::capputils::attributes::IAttribute*> cname :: attributes;     \
  const std::string cname :: ClassName = capputils::reflection::trimTypeName(typeid(cname).name());  \
  void cname::initializeClass() const { \
    static bool initialized = false; \
    if (initialized) return; \
    addAttributes(&cname::attributes, ##args, 0);
#endif
#define EndPropertyDefinitions initialized = true; }

#endif /* CAPPUTILS_REFLECTABLECLASS_H_ */
