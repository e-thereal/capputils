#pragma once
#ifndef CAPPUTILS_VARIANT_H_
#define CAPPUTILS_VARIANT_H_

#include "capputils.h"

#include <typeinfo>

namespace capputils {

class IVariant {
public:
  virtual ~IVariant();
};

template<class T>
class Variant : public IVariant {

private:
  T value;

public:
  Variant(const T& value) : value(value) { }

  T getValue() const { return value; }
  void setValue(const T& value) { this->value = value; }
};

}

#endif /* CAPPUTILS_VARIANT_H_ */