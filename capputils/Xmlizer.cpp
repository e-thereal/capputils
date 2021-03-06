/*
 * Xmlizer.cpp
 *
 *  Created on: Jan 7, 2011
 *      Author: tombr
 */

#include <capputils/Xmlizer.h>

#include <capputils/attributes/DescriptionAttribute.h>
#include <capputils/attributes/IReflectableAttribute.h>
#include <capputils/attributes/IEnumerableAttribute.h>
#include <capputils/attributes/ScalarAttribute.h>
#include <capputils/attributes/ReuseAttribute.h>
#include <capputils/attributes/VolatileAttribute.h>
#include <capputils/attributes/IXmlableAttribute.h>
#include <capputils/attributes/RenamedAttribute.h>

#include <capputils/reflection/ReflectableClassFactory.h>

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
    if (reflectable) {
      if (reflectable->getAttribute<ScalarAttribute>()) {
        propertyElement->SetAttribute("value", property->getStringValue(object));
      } else {
        propertyElement->LinkEndChild(Xmlizer::CreateXml(*reflectable));
      }
    }
  } else if (enumerableAttribute) {
    TiXmlElement* collectionElement = new TiXmlElement("Collection");
    boost::shared_ptr<IPropertyIterator> iter = enumerableAttribute->getPropertyIterator(object, property);
    while (!iter->eof()) {
      TiXmlElement* itemElement = new TiXmlElement("Item");
      populatePropertyElement(itemElement, object, iter.get());
      collectionElement->LinkEndChild(itemElement);
      iter->next();
    }
    propertyElement->LinkEndChild(collectionElement);
  } else {
    propertyElement->SetAttribute("value", property->getStringValue(object));
  }
  const vector<IAttribute*>& attributes = property->getAttributes();
  for (unsigned i = 0; i < attributes.size(); ++i) {
    IXmlableAttribute* xmlable = dynamic_cast<IXmlableAttribute*>(attributes[i]);
    if (xmlable) {
      xmlable->addToPropertyNode(*propertyElement, object, property);
    }
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

  const vector<IAttribute*>& attributes = object.getAttributes();
  for (unsigned i = 0; i < attributes.size(); ++i) {
    IXmlableAttribute* xmlable = dynamic_cast<IXmlableAttribute*>(attributes[i]);
    if (xmlable) {
      TiXmlElement* element = dynamic_cast<TiXmlElement*>(&xmlNode);
      if (element)
        xmlable->addToReflectableClassNode(*element, object);
    }
  }
}

void Xmlizer::ToXml(std::ostream& os, const reflection::ReflectableClass& object) {
  DescriptionAttribute* description = object.getAttribute<DescriptionAttribute>();
  if (description)
    ToDocument(os, Xmlizer::CreateXml(object), description->getDescription());
  else
    ToDocument(os, Xmlizer::CreateXml(object));
}

void Xmlizer::ToXml(const ::std::string& filename, const reflection::ReflectableClass& object) {
  DescriptionAttribute* description = object.getAttribute<DescriptionAttribute>();
  if (description)
    ToFile(filename, Xmlizer::CreateXml(object), description->getDescription());
  else
    ToFile(filename, Xmlizer::CreateXml(object));
}

void Xmlizer::ToFile(const ::std::string& filename, TiXmlNode* node, const std::string& description) {
  TiXmlDocument xmlDoc;
  xmlDoc.LinkEndChild(new TiXmlDeclaration("1.0", "", ""));
  if (description.size())
    xmlDoc.LinkEndChild(new TiXmlComment(description.c_str()));
  xmlDoc.LinkEndChild(node);
  xmlDoc.SaveFile(filename);
}

void Xmlizer::ToDocument(std::ostream& os, TiXmlNode* node, const std::string& description) {
  TiXmlDocument xmlDoc;
  xmlDoc.LinkEndChild(new TiXmlDeclaration("1.0", "", ""));
  if (description.size())
    xmlDoc.LinkEndChild(new TiXmlComment(description.c_str()));
  xmlDoc.LinkEndChild(node);

  TiXmlPrinter printer;
  printer.SetIndent("  ");
  xmlDoc.Accept(&printer);
  os << printer.Str();
}

ReflectableClass* Xmlizer::CreateReflectableClass(TiXmlNode& xmlNode) {
  ReflectableClass* object = ReflectableClassFactory::getInstance().newInstance(makeClassName(xmlNode.Value()));

  if (object) {
    // Check if the object has been renamed. In that case get the new name, free the old object, create the new object and rename the XmlNode
    RenamedAttribute* renamed = object->getAttribute<RenamedAttribute>();
    ReflectableClass* renamedObject = NULL;
    if (renamed) {
      try {
        renamedObject = ReflectableClassFactory::getInstance().newInstance(renamed->getNewName());
      } catch (...) { }
      if (renamedObject) {
        xmlNode.SetValue(makeXmlName(renamed->getNewName()));
        delete object;
        object = renamedObject;
      }
    }

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

void setValueOfProperty(reflection::ReflectableClass& object, reflection::IClassProperty* property, TiXmlElement* element) {

  // TODO: how to read attributes right? Need an atomic read. Setting a property might change
  //       an attribute (e.g. TimeStamps are set when the property is set). So attributes should
  //       be read after setting the value of a property, thus overriding the false attribute with
  //       the saved value. But setting a property could trigger other actions that read an attribute,
  //       thus attributes should be read before setting the value of a property. Actions are triggered
  //       through the observe mechanism. Probable solution is not to fire change events until all attributes
  //       have been read. This requires a lock and unlock mechanism for observable classes. Current workaround
  //       is to read attributes twice. Make sure, that the observe attribute is the first attribute thus
  //       guaranteeing, that no attributes are executed before the change event has been fired.
  const vector<IAttribute*>& attributes = property->getAttributes();
  for (unsigned i = 0; i < attributes.size(); ++i) {
    IXmlableAttribute* xmlable = dynamic_cast<IXmlableAttribute*>(attributes[i]);
    if (xmlable)
      xmlable->getFromPropertyNode(*element, object, property);
  }

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
      } else if (element->FirstChild()) {
        ReflectableClass* newObject = Xmlizer::CreateReflectableClass(*element->FirstChild());
        reflectable->setValuePtr(object, property, newObject);
        if (!reflectable->isPointer() && !reflectable->isSmartPointer())
          delete newObject;
      }
    } else if (enumerableAttribute) {
      TiXmlNode* collectionElement = element->FirstChild("Collection");
      boost::shared_ptr<IPropertyIterator> iter = enumerableAttribute->getPropertyIterator(object, property);
      for (TiXmlNode* node = collectionElement->FirstChild("Item"); node; node = node->NextSibling("Item")) {
        if (node->Type() == TiXmlNode::TINYXML_ELEMENT) {
          TiXmlElement* itemElement = dynamic_cast<TiXmlElement*>(node);
          setValueOfProperty(object, iter.get(), itemElement);
          iter->next();
        }
      }
    }
  }
  //attributes = property->getAttributes();
  for (unsigned i = 0; i < attributes.size(); ++i) {
    IXmlableAttribute* xmlable = dynamic_cast<IXmlableAttribute*>(attributes[i]);
    if (xmlable)
      xmlable->getFromPropertyNode(*element, object, property);
  }
}

void Xmlizer::GetPropertyFromXml(reflection::ReflectableClass& object,
      reflection::IClassProperty* property, TiXmlNode& xmlNode)
{
  using namespace std;
  TiXmlNode* xNode = &xmlNode;

  // TODO: error handling
  if (!property)
    return;

  if (makeXmlName(object.getClassName()).compare(xNode->Value()) != 0) {
    xNode = xNode->FirstChild(makeXmlName(object.getClassName()));
  }

  TiXmlNode* node = xNode->FirstChild(property->getName());
  if (node) {
    if (node->Type() == TiXmlNode::TINYXML_ELEMENT) {
      TiXmlElement* element = dynamic_cast<TiXmlElement*>(node);
      setValueOfProperty(object, property, element);
    }
  }
}

void Xmlizer::GetPropertyFromXml(reflection::ReflectableClass& object,
      reflection::IClassProperty* property, const std::string& filename)
{
  TiXmlDocument doc;
  if (!doc.LoadFile(filename)) {
    // TODO: Error handling
    return;
  }
  Xmlizer::GetPropertyFromXml(object, property, doc);
}

void Xmlizer::FromXml(reflection::ReflectableClass& object, TiXmlNode& xmlNode) {
  using namespace std;
  TiXmlNode* xNode = &xmlNode;

  if (makeXmlName(object.getClassName()).compare(xNode->Value()) != 0) {
    xNode = xNode->FirstChild(makeXmlName(object.getClassName()));
  }

  for (TiXmlNode* node = xNode->FirstChild(); node; node = node->NextSibling()) {
    if (node->Type() == TiXmlNode::TINYXML_ELEMENT) {
      TiXmlElement* element = dynamic_cast<TiXmlElement*>(node);
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

void Xmlizer::FromXmlString(reflection::ReflectableClass& object, const std::string& xmlString) {
  TiXmlDocument doc;
  doc.Parse(xmlString.c_str(), 0, TIXML_ENCODING_UTF8);
  Xmlizer::FromXml(object, doc);
}

}
