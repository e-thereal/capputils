/*
 * Executer.h
 *
 *  Created on: Aug 16, 2011
 *      Author: tombr
 */

#ifndef CAPPUTILS_EXECUTER_H_
#define CAPPUTILS_EXECUTER_H_

#include <sstream>

namespace capputils {

class Executer {
private:
  std::stringstream command;
  std::stringstream output;
  bool verbose;

public:
  Executer(bool verbose = false);
  virtual ~Executer();

  std::ostream& getCommand();
  std::string getCommandString() const;
  int execute();
  std::string getOutput() const;
};

}
#endif /* CAPPUTILS_EXECUTER_H_ */
