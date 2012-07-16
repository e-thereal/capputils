#include "Person.h"

#include <capputils/DescriptionAttribute.h>
#include <capputils/EnumeratorAttribute.h>
#include <capputils/ScalarAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/TimeStampAttribute.h>

using namespace capputils::attributes;

BeginPropertyDefinitions(Address)

DefineProperty(Street, Observe(PROPERTY_ID))
DefineProperty(City, Observe(PROPERTY_ID))
DefineProperty(StreetNumber, Observe(PROPERTY_ID))
DefineProperty(AppartmentNumber, Observe(PROPERTY_ID))

EndPropertyDefinitions

Address::Address() : _Street("W 11th Ave"), _City("Vancouver"), _StreetNumber(1065), _AppartmentNumber(207) { }

BeginPropertyDefinitions(Person)

DefineProperty(FirstName,
  Description("Persons given name."), Observe(PROPERTY_ID))

DefineProperty(Name,
  Description("Name of our parents."), Observe(PROPERTY_ID))

DefineProperty(Age,
  Description("Age in years."), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

ReflectableProperty(Address,
  Description("Address with everything."), Observe(PROPERTY_ID))

DefineProperty(Gender, Enumerator<Gender>(), Observe(PROPERTY_ID))

EndPropertyDefinitions

Person::Person(void) : _FirstName("Tom"), _Name("Brosch"), _Age(27), _Address(0), _Gender(Gender::Male)
{
  setAddress(new Address());
}

Person::~Person(void)
{
}
