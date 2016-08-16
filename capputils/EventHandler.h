#pragma once
#ifndef _CAPPUTILS_EVENTHANDLER_H_
#define _CAPPUTILS_EVENTHANDLER_H_

#include <capputils/ObservableClass.h>

namespace capputils {

class EventHandlerBase {
protected:
  std::string handlerUuid;

public:
  EventHandlerBase();
  virtual ~EventHandlerBase() { }
};

template <class T>
class EventHandler0 : public EventHandlerBase {
public:
  typedef void (T::*MemberFunc)();

private:
  T* object;
  MemberFunc method;

public:
  EventHandler0(T* object, MemberFunc method) : EventHandlerBase(), object(object), method(method) { }

  void operator()() {
    (object->*method)();
  }

  bool operator==(const EventHandler0<T>& handler) const {
    return handlerUuid.compare(handler.handlerUuid) == 0;
  }
};

template <class T, class P1>
class EventHandler1 : public EventHandlerBase {
public:
  typedef void (T::*MemberFunc)(P1 p1);

private:
  T* object;
  MemberFunc method;

public:
  EventHandler1(T* object, MemberFunc method) : EventHandlerBase(), object(object), method(method) { }

  void operator()(P1 p1) {
    (object->*method)(p1);
  }

  bool operator==(const EventHandler1<T, P1>& handler) const {
    return handlerUuid.compare(handler.handlerUuid) == 0;
  }
};

template <class T, class P1, class P2>
class EventHandler2 : public EventHandlerBase {
public:
  typedef void (T::*MemberFunc)(P1 p1, P2 p2);

private:
  T* object;
  MemberFunc method;

public:
  EventHandler2(T* object, MemberFunc method) : EventHandlerBase(), object(object), method(method) { }

  void operator()(P1 p1, P2 p2) {
    (object->*method)(p1, p2);
  }

  bool operator==(const EventHandler2<T, P1, P2>& handler) const {
    return handlerUuid.compare(handler.handlerUuid) == 0;
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
