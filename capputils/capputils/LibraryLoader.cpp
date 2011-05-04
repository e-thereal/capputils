/*
 * LibraryLoader.cpp
 *
 *  Created on: May 4, 2011
 *      Author: tombr
 */

#include "LibraryLoader.h"

#include <dlfcn.h>

using namespace std;

namespace capputils {

LibraryLoader* LibraryLoader::instance = 0;

LibraryLoader::LibraryLoader() {
}

LibraryLoader::~LibraryLoader() {
  for (map<string, LibraryData*>::iterator iter = libraryTable.begin();
      iter != libraryTable.end(); ++iter)
  {
    LibraryData* data = iter->second;
    dlclose(data->handle);
    delete data;
  }
  libraryTable.clear();
}

LibraryLoader& LibraryLoader::getInstance() {
  if (!instance)
    instance = new LibraryLoader();
  return *instance;
}

void LibraryLoader::loadLibrary(const string& filename) {
  // If loaded, increase counter, else load
  map<string, LibraryData*>::iterator iter = libraryTable.find(filename);
  if (iter == libraryTable.end()) {
    LibraryData* data = new LibraryData();
    data->filename = filename;
    data->loadCount = 1;
    data->handle = dlopen(filename.c_str(), RTLD_LAZY);
  } else {
    iter->second->loadCount = iter->second->loadCount + 1;
  }
}

void LibraryLoader::freeLibrary(const string& filename) {
  map<string, LibraryData*>::iterator iter = libraryTable.find(filename);
  if (iter != libraryTable.end()) {
    LibraryData* data = iter->second;
    data->loadCount = data->loadCount - 1;
    if (!data->loadCount) {
      dlclose(data->handle);
      libraryTable.erase(filename);
      delete data;
    }
  }
}

}
