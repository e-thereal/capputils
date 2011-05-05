/*
 * LibraryLoader.cpp
 *
 *  Created on: May 4, 2011
 *      Author: tombr
 */

#include "LibraryLoader.h"

#include <dlfcn.h>
#include <iostream>

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
    libraryTable[filename] = data;
  } else {
    iter->second->loadCount = iter->second->loadCount + 1;
    cout << filename << " library counter incremented (" << iter->second->loadCount << ")." << endl;
  }
}

void LibraryLoader::freeLibrary(const string& filename) {
  cout << "Try to unload " << filename << endl;
  map<string, LibraryData*>::iterator iter = libraryTable.find(filename);
  if (iter != libraryTable.end()) {
    LibraryData* data = iter->second;
    data->loadCount = data->loadCount - 1;
    cout << filename << " library counter decremented (" << data->loadCount << ")." << endl;
    if (!data->loadCount) {
      dlclose(data->handle);
      libraryTable.erase(filename);
      delete data;
      cout << "Library freed." << endl;
    }
  }
}

}
