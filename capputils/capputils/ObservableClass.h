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

namespace capputils {

class ObservableClass {
public:
  boost::signal<void (ObservableClass* sender, int eventId)> Changed;

public:
  ObservableClass();
  virtual ~ObservableClass();

  void fireEvent(int eventId);
};

}

#endif /* OBSERVABLECLASS_H_ */
