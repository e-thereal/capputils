/*
 * Logbook.h
 *
 *  Created on: Jul 11, 2012
 *      Author: tombr
 */

#ifndef CAPPUTILS_LOGBOOK_H_
#define CAPPUTILS_LOGBOOK_H_

#include <capputils/AbstractLogbookModel.h>

namespace capputils {

class Logbook;

class LogEntry {
private:
  boost::shared_ptr<std::stringstream> message;
  Logbook* logbook;
  Severity severity;

public:
  LogEntry(Logbook* logbook, const Severity& severity);
  virtual ~LogEntry();

  template<class T>
  std::ostream& operator<<(const T& value) {
    return *message << value;
  }

  operator std::ostream&() const {
    return *message;
  }
};

class Logbook {
private:
  AbstractLogbookModel* model;
  Severity severity;
  std::string module, uuid;

public:
  Logbook(AbstractLogbookModel* model = 0);
  virtual ~Logbook();

  LogEntry operator()();
  LogEntry operator()(const Severity& severity);

  void setModel(AbstractLogbookModel* model);
  void setSeverity(const Severity& severity);
  void setModule(const std::string& module);
  void setUuid(const std::string& uuid);

  virtual void addMessage(const std::string& message, const Severity& severity);
};

} /* namespace capputils */

#endif /* CAPPUTILS_LOGBOOK_H_ */
