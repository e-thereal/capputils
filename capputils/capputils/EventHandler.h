#pragma once
#ifndef _CAPPUTILS_EVENTHANDLER_H_
#define _CAPPUTILS_EVENTHANDLER_H_

#include "ObservableClass.h"

namespace capputils {

template <class T>
class EventHandler0 {
public:
  typedef void (T::*MemberFunc)();

private:
  T* object;
  MemberFunc method;

public:
  EventHandler0(T* object, MemberFunc method) : object(object), method(method) { }

  void operator()() {
    (object->*method)();
  }
};

template <class T, class P1>
class EventHandler1 {
public:
  typedef void (T::*MemberFunc)(P1 p1);

private:
  T* object;
  MemberFunc method;

public:
  EventHandler1(T* object, MemberFunc method) : object(object), method(method) { }

  void operator()(P1 p1) {
    (object->*method)(p1);
  }
};

template <class T, class P1, class P2>
class EventHandler2 {
public:
  typedef void (T::*MemberFunc)(P1 p1, P2 p2);

private:
  T* object;
  MemberFunc method;

public:
  EventHandler2(T* object, MemberFunc method) : object(object), method(method) { }

  void operator()(P1 p1, P2 p2) {
    (object->*method)(p1, p2);
  }
};

template <class T>
class EventHandler : public EventHandler2<T, ObservableClass*, int> {
typedef EventHandler2<T, ObservableClass*, int> Base;
public:
  EventHandler(T* object, typename Base::MemberFunc method)
  : Base(object, method) { }
};

/*template <class T>
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
};*/

}

#endif
