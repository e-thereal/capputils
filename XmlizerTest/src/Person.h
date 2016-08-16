#pragma once

#ifndef _PERSON_H_
#define _PERSON_H_

#include <capputils/reflection/ReflectableClass.h>
#include <string>
#include <istream>
#include <ostream>
#include <capputils/Enumerators.h>
#include <capputils/ObservableClass.h>
#include <capputils/TimedClass.h>

class Address : public capputils::reflection::ReflectableClass,
                public capputils::ObservableClass
{

InitReflectableClass(Address)

Property(Street, std::string)
Property(City, std::string)
Property(StreetNumber, int)
Property(AppartmentNumber, int)

public:
  Address();
  virtual ~Address() { }

  virtual void toStream(std::ostream& stream) const {
    stream << getAppartmentNumber() << "-" << getStreetNumber() << " " << getStreet() << "; " << getCity();
  }

  virtual void fromStream(std::istream& stream) {
    int appartNum, streetNum;
    char ch;
    stream >> appartNum >> ch >> streetNum;
    _AppartmentNumber = appartNum;
    _StreetNumber = streetNum;
    std::string str, street;
    do {
      stream >> str;
      street.append(str);
      street.append(" ");
    } while (str[str.size()-1] != ';');
    _Street = street.substr(0, street.size()-2);
    stream >> _City;
  }
};

CapputilsEnumerator(Gender, Male, Female, Neutrum);

class Person : public capputils::reflection::ReflectableClass,
               public capputils::ObservableClass,
               public capputils::TimedClass
{
InitReflectableClass(Person)

Property(FirstName, std::string)
Property(Name, std::string)
Property(Age, int)
Property(Address, boost::shared_ptr<Address>)
Property(Gender, Gender)

public:
  Person(void);
  virtual ~Person(void);
};

#endif
