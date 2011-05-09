/*
 * LibraryLoader.h
 *
 *  Created on: May 4, 2011
 *      Author: tombr
 */

#ifndef LIBRARYLOADER_H_
#define LIBRARYLOADER_H_

#include <string>
#include <map>

#include <boost/signals.hpp>
#include "EventHandler.h"

namespace capputils {

#ifdef _WIN32
struct HandleWrapper;
#endif

struct LibraryData {
  std::string filename;
  time_t lastModified;
  int loadCount;
#ifdef _WIN32
  HandleWrapper* handleWrapper;
#else
  void* handle;
#endif
  LibraryData(const char* filename);
  virtual ~LibraryData();
};

class LibraryLoader {
private:
  static LibraryLoader* instance;
  std::map<std::string, LibraryData*> libraryTable;

protected:
  LibraryLoader();

public:
  virtual ~LibraryLoader();

  static LibraryLoader& getInstance();

  void loadLibrary(const std::string& filename);
  void freeLibrary(const std::string& filename);
  bool librariesUpdated();
};

}

#endif /* LIBRARYLOADER_H_ */
