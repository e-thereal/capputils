/*
 * Xmlizer.cpp
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#include "Xmlizer.h"

#include "DescriptionAttribute.h"

#include <iostream>

namespace capputils {

using namespace attributes;

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
    propertyElement->SetAttribute("value", properties[i]->getStringValue(object));
    xmlNode.LinkEndChild(propertyElement);
  }
}

void Xmlizer::ToXml(const ::std::string& filename, const reflection::ReflectableClass& object) {
  TiXmlDocument xmlDoc;
  xmlDoc.LinkEndChild(new TiXmlDeclaration("1.0", "", ""));
  xmlDoc.LinkEndChild(Xmlizer::CreateXml(object));
  xmlDoc.SaveFile(filename);
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
      const char* value = element->Attribute("value");
      if (value) {
        reflection::IClassProperty* property = object.findProperty(element->Value());
        if (property)
          property->setStringValue(object, value);
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
