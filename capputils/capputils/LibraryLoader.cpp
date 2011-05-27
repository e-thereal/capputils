/*
 * LibraryLoader.cpp
 *
 *  Created on: May 4, 2011
 *      Author: tombr
 */

#include "LibraryLoader.h"

#include <set>

#ifndef _WIN32
#include <dlfcn.h>
#else
#include <Windows.h>
#define BOOST_FILESYSTEM_VERSION 3
#endif
#include <iostream>
#include <boost/filesystem.hpp>
#include "LibraryException.h"
#include "ReflectableClassFactory.h"

using namespace std;
using namespace boost::filesystem;

namespace capputils {

#ifdef _WIN32
struct HandleWrapper {
  HMODULE handle;
};

LibraryData::LibraryData(const char* filename) {
  // TODO: Error handling, can't copy or can't load + why can't load

  reflection::ReflectableClassFactory& factory = reflection::ReflectableClassFactory::getInstance();
  set<string> loadedClasses;
  vector<string>& loadedClassNames = factory.getClassNames();
  for (unsigned i = 0; i < loadedClassNames.size(); ++i)
    loadedClasses.insert(loadedClassNames[i]);

  handleWrapper = new HandleWrapper();
  this->filename = filename;
  string tmpName = this->filename + ".host_copy.dll";
  copy_file(filename, tmpName.c_str(), copy_option::overwrite_if_exists);
  loadCount = 1;
  handleWrapper->handle = LoadLibraryA(tmpName.c_str());
  if (!handleWrapper->handle)
    throw exceptions::LibraryException();
  lastModified = last_write_time(filename);

  loadedClassNames = factory.getClassNames();
  for (unsigned i = 0; i < loadedClassNames.size(); ++i) {
    if (loadedClasses.find(loadedClassNames[i]) == loadedClasses.end())
      classnames.push_back(loadedClassNames[i]);
  }
}

LibraryData::~LibraryData() {
  FreeLibrary(handleWrapper->handle);
  delete handleWrapper;

  string tmpName = this->filename + ".host_copy.dll";
  remove(tmpName.c_str());
}

#else

LibraryData::LibraryData(const char* filename) {
  // Get classnames before and after loading the library
  // Diff are all classes that come with the library
  reflection::ReflectableClassFactory& factory = reflection::ReflectableClassFactory::getInstance();
  set<string> loadedClasses;
  vector<string>& loadedClassNames = factory.getClassNames();
  for (unsigned i = 0; i < loadedClassNames.size(); ++i)
    loadedClasses.insert(loadedClassNames[i]);

  this->filename = filename;
  loadCount = 1;
  handle = dlopen(filename, RTLD_NOW);
  if (!handle)
    throw exceptions::LibraryException();
  lastModified = last_write_time(filename);

  loadedClassNames = factory.getClassNames();
  for (unsigned i = 0; i < loadedClassNames.size(); ++i) {
    if (loadedClasses.find(loadedClassNames[i]) == loadedClasses.end())
      classnames.push_back(loadedClassNames[i]);
  }
}

LibraryData::~LibraryData() {
  //cout << "Unloading library: " << filename << endl;
  dlclose(handle);
}

#endif

LibraryLoader* LibraryLoader::instance = 0;

LibraryLoader::LibraryLoader() {
}

LibraryLoader::~LibraryLoader() {
  for (map<string, LibraryData*>::iterator iter = libraryTable.begin();
      iter != libraryTable.end(); ++iter)
  {
    LibraryData* data = iter->second;
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
    LibraryData* data = new LibraryData(filename.c_str());
    libraryTable[filename] = data;
    //cout << filename << " library loaded." << endl;
  } else {
    iter->second->loadCount = iter->second->loadCount + 1;
    //cout << filename << " library counter incremented (" << iter->second->loadCount << ")." << endl;
  }
}

void LibraryLoader::freeLibrary(const string& filename) {
  //cout << "Try to free library " << filename << endl;
  map<string, LibraryData*>::iterator iter = libraryTable.find(filename);
  if (iter != libraryTable.end()) {
    LibraryData* data = iter->second;
    data->loadCount = data->loadCount - 1;
    //cout << filename << " library counter decremented (" << data->loadCount << ")." << endl;
    if (!data->loadCount) {
      libraryTable.erase(filename);
      delete data;
      //cout << "Library freed." << endl;
    }
  }
}

bool LibraryLoader::librariesUpdated() {
  bool updated = false;
  time_t lastModified = 0;
  for (map<string, LibraryData*>::iterator iter = libraryTable.begin();
      iter != libraryTable.end(); ++iter)
  {
    LibraryData* data = iter->second;
    try {
      lastModified = last_write_time(data->filename);
    } catch(...) {
      continue;
    }
    if (lastModified != data->lastModified) {
      data->lastModified = lastModified;
      updated = true;
    }
  }
  return updated;
}

string LibraryLoader::classDefinedIn(const string& classname) {
  for (map<string, LibraryData*>::iterator iter = libraryTable.begin();
        iter != libraryTable.end(); ++iter)
  {
    LibraryData* data = iter->second;
    for (unsigned i = 0; i < data->classnames.size(); ++i) {
      if (data->classnames[i].compare(classname) == 0)
        return iter->first;
    }
  }

  return "";
}

}
