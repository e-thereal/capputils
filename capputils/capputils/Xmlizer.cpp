/*
 * Xmlizer.cpp
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#include "Xmlizer.h"

#include "DescriptionAttribute.h"
#include "IReflectableAttribute.h"
#include "ScalarAttribute.h"
#include "ReflectableClassFactory.h"

#include <iostream>

namespace capputils {

using namespace attributes;
using namespace reflection;

TiXmlElement* Xmlizer::CreateXml(const reflection::ReflectableClass& object) {
  TiXmlElement* xmlNode = new TiXmlElement(object.getClassName());

  Xmlizer::ToXml(*xmlNode, object);
  return xmlNode;
}

void Xmlizer::ToXml(TiXmlNode& xmlNode, const reflection::ReflectableClass& object) {
  std::vector<reflection::IClassProperty*>& properties = object.getProperties();

  for (unsigned i = 0; i < properties.size(); ++i) {
    DescriptionAttribute* description = properties[i]->getAttribute<DescriptionAttribute>();
    if (description) {
      xmlNode.LinkEndChild(new TiXmlComment(description->getDescription().c_str()));
    }

    TiXmlElement* propertyElement = new TiXmlElement(properties[i]->getName());
    IReflectableAttribute* reflectableAttribute = properties[i]->getAttribute<IReflectableAttribute>();
    if (reflectableAttribute) {
      ReflectableClass* reflectable = reflectableAttribute->getValuePtr(object, properties[i]);
      if (reflectable->getAttribute<ScalarAttribute>()) {
        propertyElement->SetAttribute("value", properties[i]->getStringValue(object));
      } else {
        propertyElement->LinkEndChild(Xmlizer::CreateXml(*reflectable));
      }
    } else {
      propertyElement->SetAttribute("value", properties[i]->getStringValue(object));
    }
    xmlNode.LinkEndChild(propertyElement);
  }
}

void Xmlizer::ToXml(const ::std::string& filename, const reflection::ReflectableClass& object) {
  TiXmlDocument xmlDoc;
  xmlDoc.LinkEndChild(new TiXmlDeclaration("1.0", "", ""));
  xmlDoc.LinkEndChild(Xmlizer::CreateXml(object));
  xmlDoc.SaveFile(filename);
}

ReflectableClass* Xmlizer::CreateReflectableClass(const TiXmlNode& xmlNode) {
  ReflectableClass* object = ReflectableClassFactory::getInstance().newInstance(xmlNode.Value());
  if (object) {
    Xmlizer::FromXml(*object, xmlNode);
    return object;
  }
  return 0;
}

void Xmlizer::FromXml(reflection::ReflectableClass& object, const TiXmlNode& xmlNode) {
  using namespace std;
  const TiXmlNode* xNode = &xmlNode;

  if (object.getClassName().compare(xNode->Value()) != 0) {
    xNode = xNode->FirstChild(object.getClassName());
  }

  for (const TiXmlNode* node = xNode->FirstChild(); node; node = node->NextSibling()) {
    if (node->Type() == TiXmlNode::TINYXML_ELEMENT) {
      const TiXmlElement* element = dynamic_cast<const TiXmlElement*>(node);
      reflection::IClassProperty* property = object.findProperty(element->Value());
      if (property) {
        const char* value = element->Attribute("value");
        if (value) {
            property->setStringValue(object, value);
        } else {
          IReflectableAttribute* reflectable = property->getAttribute<IReflectableAttribute>();
          if (reflectable) {
            reflectable->setValuePtr(object, property, Xmlizer::CreateReflectableClass(*node->FirstChild()));
          }
        }
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
