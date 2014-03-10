/*
 * AbstractLogbook.h
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#ifndef CAPPUTILS_ABSTRACTLOGBOOK_H_
#define CAPPUTILS_ABSTRACTLOGBOOK_H_

#include <capputils/capputils.h>

#include <boost/shared_ptr.hpp>
#include <sstream>

#include <capputils/Enumerators.h>

namespace capputils {

CapputilsEnumerator(Severity, Trace, Message, Warning, Error);

class AbstractLogbookModel {
public:
  virtual ~AbstractLogbookModel();

  virtual void addMessage(const std::string& message,
      const Severity& severity = Severity::Message,
      const std::string& module = "<none>",
      const std::string& uuid = "<none>") = 0;
};

} /* namespace capputils */

#endif /* ABSTRACTLOGBOOK_H_ */
