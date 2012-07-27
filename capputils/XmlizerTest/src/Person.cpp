#include "Person.h"

#include <capputils/DescriptionAttribute.h>
#include <capputils/EnumeratorAttribute.h>
#include <capputils/ScalarAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/TimeStampAttribute.h>

using namespace capputils::attributes;

BeginPropertyDefinitions(Address)

DefineProperty(Street, Observe(Id))
DefineProperty(City, Observe(Id))
DefineProperty(StreetNumber, Observe(Id))
DefineProperty(AppartmentNumber, Observe(Id))

EndPropertyDefinitions

Address::Address() : _Street("W 11th Ave"), _City("Vancouver"), _StreetNumber(1065), _AppartmentNumber(207) { }

BeginPropertyDefinitions(Person)

DefineProperty(FirstName,
  Description("Persons given name."), Observe(Id))

DefineProperty(Name,
  Description("Name of our parents."), Observe(Id))

DefineProperty(Age,
  Description("Age in years."), Observe(Id), TimeStamp(Id))

ReflectableProperty(Address,
  Description("Address with everything."), Observe(Id))

DefineProperty(Gender, Enumerator<Gender>(), Observe(Id))

EndPropertyDefinitions

Person::Person(void) : _FirstName("Tom"), _Name("Brosch"), _Age(27), _Address(0), _Gender(Gender::Male)
{
  setAddress(new Address());
}

Person::~Person(void)
{
}
