/*
 * IAttribute.h
 *
 *  Created on: Feb 10, 2011
 *      Author: tombr
 */

#ifndef IATTRIBUTE_H_
#define IATTRIBUTE_H_

namespace capputils {

namespace attributes {

class IAttribute {
public:
  virtual ~IAttribute();
};

class AttributeWrapper {
public:
  IAttribute* attribute;

  AttributeWrapper(IAttribute* attribute) : attribute (attribute) { }
};

}

}

#endif /* IATTRIBUTE_H_ */
