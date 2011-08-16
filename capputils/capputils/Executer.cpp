/*
 * Executer.cpp
 *
 *  Created on: Aug 16, 2011
 *      Author: tombr
 */

#include "Executer.h"

#include <cstdio>

#ifdef WIN32
#define popen   _popen
#define pclose _pclose
#endif

namespace capputils {

Executer::Executer() {
}

Executer::~Executer() {
}

std::ostream& Executer::getCommand() {
  return command;
}

int Executer::execute() {
  int ch;
  FILE* stream = popen(command.str().c_str(), "r");
  if (stream) {
    while ((ch=fgetc(stream)) != EOF ) {
      output << (char)ch;
    }
    pclose(stream);
  } else {
    return 1;
  }

  return 0;
}

std::string Executer::getOutput() const {
  return output.str();
}

} /* namespace capputils */
