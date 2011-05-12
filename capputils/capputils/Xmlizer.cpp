/*
 * Xmlizer.cpp
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#include "Xmlizer.h"

#include "DescriptionAttribute.h"
#include "IReflectableAttribute.h"
#include "IEnumerableAttribute.h"
#include "ScalarAttribute.h"
#include "ReflectableClassFactory.h"
#include "ReuseAttribute.h"
#include "VolatileAttribute.h"

#include <iostream>

using namespace std;

namespace capputils {

using namespace attributes;
using namespace reflection;

string replaceAll(const string& context, const string& from, const string& to)
{
  string str = context;
  size_t lookHere = 0;
  size_t foundHere;
  while((foundHere = str.find(from, lookHere)) != string::npos)
  {
    str.replace(foundHere, from.size(), to);
        lookHere = foundHere + to.size();
  }
  return str;
}

string makeXmlName(const string& className) {
  return replaceAll(className.substr(), "::", "-");
}

string makeClassName(const string& xmlName) {
  return replaceAll(xmlName.substr(), "-", "::");
}

TiXmlElement* Xmlizer::CreateXml(const reflection::ReflectableClass& object) {
  TiXmlElement* xmlNode = new TiXmlElement(makeXmlName(object.getClassName()));

  Xmlizer::ToXml(*xmlNode, object);
  return xmlNode;
}

void populatePropertyElement(TiXmlElement* propertyElement, const ReflectableClass& object, const IClassProperty* property) {
  IReflectableAttribute* reflectableAttribute = property->getAttribute<IReflectableAttribute>();
  IEnumerableAttribute* enumerableAttribute = property->getAttribute<IEnumerableAttribute>();
  if (reflectableAttribute) {
    ReflectableClass* reflectable = reflectableAttribute->getValuePtr(object, property);
    if (reflectable->getAttribute<ScalarAttribute>()) {
      propertyElement->SetAttribute("value", property->getStringValue(object));
    } else {
      propertyElement->LinkEndChild(Xmlizer::CreateXml(*reflectable));
    }
  } else if (enumerableAttribute) {
    TiXmlElement* collectionElement = new TiXmlElement("Collection");
    IPropertyIterator* iter = enumerableAttribute->getPropertyIterator(property);
    while (!iter->eof(object)) {
      TiXmlElement* itemElement = new TiXmlElement("Item");
      populatePropertyElement(itemElement, object, iter);
      collectionElement->LinkEndChild(itemElement);
      iter->next();
    }
    propertyElement->LinkEndChild(collectionElement);
    delete iter;
  } else {
    propertyElement->SetAttribute("value", property->getStringValue(object));
  }
}

void Xmlizer::AddPropertyToXml(TiXmlNode& xmlNode, const ReflectableClass& object,
      const IClassProperty* property)
{
  DescriptionAttribute* description = property->getAttribute<DescriptionAttribute>();
  if (description) {
    xmlNode.LinkEndChild(new TiXmlComment(description->getDescription().c_str()));
  }

  TiXmlElement* propertyElement = new TiXmlElement(property->getName());
  populatePropertyElement(propertyElement, object, property);
  xmlNode.LinkEndChild(propertyElement);
}

void Xmlizer::ToXml(TiXmlNode& xmlNode, const ReflectableClass& object) {
  std::vector<reflection::IClassProperty*>& properties = object.getProperties();

  for (unsigned i = 0; i < properties.size(); ++i) {
    if (properties[i]->getAttribute<VolatileAttribute>())
      continue;

    AddPropertyToXml(xmlNode, object, properties[i]);
  }
}

void Xmlizer::ToXml(const ::std::string& filename, const reflection::ReflectableClass& object) {
  ToFile(filename, Xmlizer::CreateXml(object));
}

void Xmlizer::ToFile(const ::std::string& filename, TiXmlNode* node) {
  TiXmlDocument xmlDoc;
  xmlDoc.LinkEndChild(new TiXmlDeclaration("1.0", "", ""));
  xmlDoc.LinkEndChild(node);
  xmlDoc.SaveFile(filename);
}

ReflectableClass* Xmlizer::CreateReflectableClass(const TiXmlNode& xmlNode) {
  ReflectableClass* object = ReflectableClassFactory::getInstance().newInstance(makeClassName(xmlNode.Value()));
  if (object) {
    Xmlizer::FromXml(*object, xmlNode);
    return object;
  }
  return 0;
}

reflection::ReflectableClass* Xmlizer::CreateReflectableClass(const ::std::string& filename) {
  TiXmlDocument doc;
  if (!doc.LoadFile(filename)) {
    // TODO: Error handling
    return 0;
  }
  return CreateReflectableClass(*doc.FirstChildElement());
}

void setValueOfProperty(reflection::ReflectableClass& object, reflection::IClassProperty* property, const TiXmlElement* element) {
  const char* value = element->Attribute("value");
  if (value) {
      property->setStringValue(object, value);
  } else {
    IReflectableAttribute* reflectable = property->getAttribute<IReflectableAttribute>();
    IEnumerableAttribute* enumerableAttribute = property->getAttribute<IEnumerableAttribute>();
    if (reflectable) {
      if (property->getAttribute<ReuseAttribute>() && reflectable->isPointer()) {
        ReflectableClass* oldObject = reflectable->getValuePtr(object, property);
        Xmlizer::FromXml(*oldObject, *element->FirstChild());
        reflectable->setValuePtr(object, property, oldObject);
      } else { 
        ReflectableClass* newObject = Xmlizer::CreateReflectableClass(*element->FirstChild());
        reflectable->setValuePtr(object, property, newObject);
        if (!reflectable->isPointer())
          delete newObject;
      }
    } else if (enumerableAttribute) {
      const TiXmlNode* collectionElement = element->FirstChild("Collection");
      IPropertyIterator* iter = enumerableAttribute->getPropertyIterator(property);
      for (const TiXmlNode* node = collectionElement->FirstChild("Item"); node; node = node->NextSibling("Item")) {
        if (node->Type() == TiXmlNode::TINYXML_ELEMENT) {
          const TiXmlElement* itemElement = dynamic_cast<const TiXmlElement*>(node);
          setValueOfProperty(object, iter, itemElement);
          iter->next();
        }
      }
      delete iter;
    }
  }
}

void Xmlizer::FromXml(reflection::ReflectableClass& object, const TiXmlNode& xmlNode) {
  using namespace std;
  const TiXmlNode* xNode = &xmlNode;

  if (makeXmlName(object.getClassName()).compare(xNode->Value()) != 0) {
    xNode = xNode->FirstChild(makeXmlName(object.getClassName()));
  }

  for (const TiXmlNode* node = xNode->FirstChild(); node; node = node->NextSibling()) {
    if (node->Type() == TiXmlNode::TINYXML_ELEMENT) {
      const TiXmlElement* element = dynamic_cast<const TiXmlElement*>(node);
      reflection::IClassProperty* property = object.findProperty(element->Value());
      if (property) {
        setValueOfProperty(object, property, element);
      }
    }
  }
}

void Xmlizer::FromXml(reflection::ReflectableClass& object, const ::std::string& filename) {
  TiXmlDocument doc;
  if (!doc.LoadFile(filename)) {
    // TODO: Error handling
    return;
  }
  Xmlizer::FromXml(object, doc);
}

}
