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

public:
  Executer();
  virtual ~Executer();

  std::ostream& getCommand();
  int execute();
  std::string getOutput() const;
};

}
#endif /* CAPPUTILS_EXECUTER_H_ */
