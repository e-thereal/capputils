/*
 * PropertyMap.cpp
 *
 *  Created on: Jul 25, 2012
 *      Author: tombr
 */

#include <capputils/PropertyMap.h>

#include <capputils/reflection/ReflectableClass.h>

namespace capputils {

using namespace reflection;

PropertyMap::PropertyMap(const reflection::ReflectableClass& object) {
  std::vector<IClassProperty*>& properties = object.getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    IClassProperty& prop = *properties[i];
    values[prop.getName()] = prop.toVariant(object);
  }
}

PropertyMap::~PropertyMap() {
  for (std::map<std::string, IVariant*>::iterator i = values.begin();
      i != values.end(); ++i)
  {
    delete i->second;
  }
  values.clear();
}

void PropertyMap::writeToReflectableClass(reflection::ReflectableClass& object) {
  std::vector<IClassProperty*>& properties = object.getProperties();
  for (unsigned i = 0; i < properties.size(); ++i) {
    IClassProperty& prop = *properties[i];
    map_t::iterator pos = values.find(prop.getName());
    if (pos != values.end()) {
      prop.fromVariant(*pos->second, object);
    }
  }
}

} /* namespace capputils */
