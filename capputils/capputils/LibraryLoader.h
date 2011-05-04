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

namespace capputils {

struct LibraryData {
  std::string filename;
  int loadCount;
  void* handle;
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
};

}

#endif /* LIBRARYLOADER_H_ */
