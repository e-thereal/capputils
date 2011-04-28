#pragma once
#ifndef _CAPPUTILS_EVENTHANDLER_H_
#define _CAPPUTILS_EVENTHANDLER_H_

#include "ObservableClass.h"

namespace capputils {

template <class T>
class EventHandler {
public:
  typedef void (T::*MemberFunc)(ObservableClass* sender, int eventId);

private:
  T* object;
  MemberFunc method;

public:
  EventHandler(T* object, MemberFunc method) : object(object), method(method) { }

  void operator()(ObservableClass* sender, int eventId) {
    (object->*method)(sender, eventId);
  }
};

}

#endif