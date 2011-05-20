/*
 * TimeStampAttribute.h
 *
 *  Created on: May 19, 2011
 *      Author: tombr
 */

#ifndef TIMESTAMPATTRIBUTE_H_
#define TIMESTAMPATTRIBUTE_H_

#include "IExecutableAttribute.h"
#include "IXmlableAttribute.h"

namespace capputils {

namespace attributes {

class TimeStampAttribute : public virtual IExecutableAttribute,
                           public virtual IXmlableAttribute
{
private:
  int propertyId;
  time_t time;

public:
  // use this header for class attributes
  TimeStampAttribute(const std::string& timeStamp);

  // Use this constructor for property attributes
  TimeStampAttribute(int propertyId);
  virtual ~TimeStampAttribute();

  time_t getTime(const reflection::ReflectableClass& object) const;
  void setTime(reflection::ReflectableClass& object, time_t time) const;
  void setTime(reflection::ReflectableClass& object, const char* timeString) const;

  virtual void executeBefore(reflection::ReflectableClass& object, const reflection::IClassProperty& property) const;
  virtual void executeAfter(reflection::ReflectableClass& object, const reflection::IClassProperty& property) const;
  virtual void addToPropertyNode(TiXmlElement& element,
      const reflection::ReflectableClass& object,
      const reflection::IClassProperty* property) const;

  virtual void getFromPropertyNode(const TiXmlElement& node,
      reflection::ReflectableClass& object,
      reflection::IClassProperty* property) const;
};

AttributeWrapper* TimeStamp(int propertyId);

}

}

#endif /* TIMESTAMPATTRIBUTE_H_ */
