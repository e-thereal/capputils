/*
 * Executer.cpp
 *
 *  Created on: Aug 16, 2011
 *      Author: tombr
 */

#include "Executer.h"

#include <cstdio>
#include <iostream>

#ifdef WIN32
#define popen   _popen
#define pclose _pclose
#endif

namespace capputils {

Executer::Executer(int verbose) : verbose(verbose) {
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
      if (verbose)
        std::cout << (char)ch;
    }
    pclose(stream);
  } else {
    return 1;
  }

  return 0;
}

std::string Executer::getCommandString() const {
  return command.str();
}

std::string Executer::getOutput() const {
  return output.str();
}

} /* namespace capputils */
