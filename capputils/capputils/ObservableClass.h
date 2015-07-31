/*
 * ObservableClass.h
 *
 *  Created on: Mar 2, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_OBSERVABLECLASS_H_
#define CAPPUTILS_OBSERVABLECLASS_H_

#include <boost/signals2/signal.hpp>
#include <capputils/reflection/IClassProperty.h>

#include <set>

namespace capputils {

class ObservableClass {
public:
  boost::signals2::signal<void (ObservableClass* sender, int eventId)> Changed;

private:
  std::map<ObservableClass*, int> parents;
  std::set<ObservableClass*> children;

public:
  ObservableClass();
  virtual ~ObservableClass();

  void connectHandler(void (*handler)(ObservableClass* sender, int eventId));
  void fireChangeEvent(int eventId);
  void addChild(ObservableClass* child, int eventId);
  void removeChild(ObservableClass* child);

protected:
  void addParent(ObservableClass* parent, int eventId);
  void removeParent(ObservableClass* parent);
};

}

#endif /* CAPPUTILS_OBSERVABLECLASS_H_ */
