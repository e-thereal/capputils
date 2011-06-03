/*
 * TimeStampAttribute.cpp
 *
 *  Created on: May 19, 2011
 *      Author: tombr
 */

#include "TimeStampAttribute.h"

#include "TimedClass.h"

#include <iostream>
#include <cstdio>

namespace capputils {

using namespace reflection;

namespace attributes {

const char * const shortMonthNames[] = {
 "Jan", "Feb", "Mar", "Apr", "May", "Jun",
 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };

time_t fromString(const std::string& str) {
  if (str.size() < 10)
    return 0;
  const char* cstr = str.c_str();
  int month = 0, day, year, hour, minute, second;
  for (unsigned i = 0; i < 12; ++i) {
    if (str.compare(0, 3, shortMonthNames[i]) == 0) {
      month = i;
      break;
    }
  }
  sscanf(cstr+4, "%d %d %d:%d:%d", &day, &year, &hour, &minute, &second);
  time_t rawtime;
  struct tm *timeinfo;
  time(&rawtime);
  timeinfo = localtime(&rawtime);

  timeinfo->tm_year = year - 1900;
  timeinfo->tm_mon = month;
  timeinfo->tm_mday = day;
  timeinfo->tm_hour = hour;
  timeinfo->tm_min = minute;
  timeinfo->tm_sec = second;

  return mktime(timeinfo);
}

TimeStampAttribute::TimeStampAttribute(const std::string& timeStamp) : propertyId(-1) {
  time = fromString(timeStamp);
}

TimeStampAttribute::TimeStampAttribute(int propertyId) : time(0), propertyId(propertyId) {
}

TimeStampAttribute::~TimeStampAttribute() {
}

time_t TimeStampAttribute::getTime(const reflection::ReflectableClass& object) const {
  const TimedClass* timed = dynamic_cast<const TimedClass*>(&object);
  if (timed)
    return timed->getTime(propertyId);
  return 0;
}

void TimeStampAttribute::setTime(reflection::ReflectableClass& object, time_t time) const {
  TimedClass* timed = dynamic_cast<TimedClass*>(&object);
  if (timed) {
    timed->setTime(propertyId, time);
  }
}

void TimeStampAttribute::setTime(reflection::ReflectableClass& object, const char* timeString) const {
  setTime(object, fromString(timeString));
}

void TimeStampAttribute::executeBefore(ReflectableClass& object, const IClassProperty& property) const {
}

void TimeStampAttribute::executeAfter(ReflectableClass& object, const IClassProperty& property) const {
  setTime(object, std::time(0));
}

void TimeStampAttribute::addToPropertyNode(TiXmlElement& node, const ReflectableClass& object,
      const IClassProperty* property) const
{
  struct tm* timeinfo;
  char buffer[256];
  time_t time = getTime(object);

  timeinfo = localtime(&time);
  strftime(buffer, 256, "%b %d %Y %H:%M:%S", timeinfo);
  node.SetAttribute("timestamp", buffer);
}

void TimeStampAttribute::getFromPropertyNode(const TiXmlElement& node, ReflectableClass& object,
      IClassProperty* property) const
{
  const char* value = node.Attribute("timestamp");
  if (value) {
    setTime(object, value);
  }
}

AttributeWrapper* TimeStamp(int propertyId) {
  return new AttributeWrapper(new TimeStampAttribute(propertyId));
}

}

}
