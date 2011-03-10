/*
 * ObservableClass.h
 *
 *  Created on: Mar 2, 2011
 *      Author: tombr
 */

#ifndef OBSERVABLECLASS_H_
#define OBSERVABLECLASS_H_

#include <boost/signal.hpp>
#include "IClassProperty.h"

#include <set>

namespace capputils {

class ObservableClass {
public:
  boost::signal<void (ObservableClass* sender, int eventId)> Changed;

private:
  std::map<ObservableClass*, int> parents;
  std::set<ObservableClass*> children;

public:
  ObservableClass();
  virtual ~ObservableClass();

  void fireEvent(int eventId);
  void addChild(ObservableClass* child, int eventId);
  void removeChild(ObservableClass* child);

protected:
  void addParent(ObservableClass* parent, int eventId);
  void removeParent(ObservableClass* parent);
};

}

#endif /* OBSERVABLECLASS_H_ */
