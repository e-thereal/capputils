/*
 * LibraryLoader.cpp
 *
 *  Created on: May 4, 2011
 *      Author: tombr
 */

#include <capputils/LibraryLoader.h>

#include <set>

#ifndef _WIN32
#include <dlfcn.h>
#else
#include <Windows.h>
#include <strsafe.h>
#endif
#include <iostream>
#include <boost/filesystem.hpp>
#include <capputils/exceptions/LibraryException.h>
#include <capputils/reflection/ReflectableClassFactory.h>

using namespace std;
using namespace boost::filesystem;

namespace capputils {

#ifdef _WIN32

void showError(LPTSTR lpszFunction) 
{ 
    // Retrieve the system error message for the last-error code

    LPVOID lpMsgBuf;
    LPVOID lpDisplayBuf;
    DWORD dw = GetLastError(); 

    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | 
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dw,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR) &lpMsgBuf,
        0, NULL );

    // Display the error message and exit the process

    lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT, 
        (lstrlen((LPCTSTR)lpMsgBuf) + lstrlen((LPCTSTR)lpszFunction) + 40) * sizeof(TCHAR)); 
    StringCchPrintf((LPTSTR)lpDisplayBuf, 
        LocalSize(lpDisplayBuf) / sizeof(TCHAR),
        TEXT("%s failed with error %d: %s"), 
        lpszFunction, dw, lpMsgBuf); 
    MessageBox(NULL, (LPCTSTR)lpDisplayBuf, TEXT("Error"), MB_OK); 

    LocalFree(lpMsgBuf);
    LocalFree(lpDisplayBuf);
}

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
  if (!handleWrapper->handle) {
    showError(TEXT("LoadLibrary"));
    throw exceptions::LibraryException(filename, "Unknown load error.");
  }
  lastModified = last_write_time(filename);

  loadedClassNames = factory.getClassNames();
  for (unsigned i = 0; i < loadedClassNames.size(); ++i) {
    if (loadedClasses.find(loadedClassNames[i]) == loadedClasses.end())
      classnames.push_back(loadedClassNames[i]);
  }
}

LibraryData::~LibraryData() { }

void LibraryData::unload() {
  // TODO: Error handling: why can't free.
  if (!FreeLibrary(handleWrapper->handle)) {
    delete handleWrapper;
    throw exceptions::LibraryException(filename, "Unknown free error.");
  }
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
  // RTLD_GLOBAL: needed to get dynamic_casts right (http://stackoverflow.com/questions/2351786/dynamic-cast-fails-when-used-with-dlopen-dlsym)
  // RTLD_GLOBAL: http://gcc.gnu.org/ml/gcc-help/2008-11/msg00174.html
  handle = dlopen(filename, RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    throw exceptions::LibraryException(filename, dlerror());
  }
  lastModified = last_write_time(filename);

  loadedClassNames = factory.getClassNames();
  for (unsigned i = 0; i < loadedClassNames.size(); ++i) {
    if (loadedClasses.find(loadedClassNames[i]) == loadedClasses.end())
      classnames.push_back(loadedClassNames[i]);
  }
}

LibraryData::~LibraryData() { }

void LibraryData::unload() {
//  cout << "Unloading library: " << filename << endl;
  if (dlclose(handle))
    throw exceptions::LibraryException(filename, dlerror());
}

#endif

LibraryLoader* LibraryLoader::instance = 0;

LibraryLoader::LibraryLoader() : autoUnload(true) { }

LibraryLoader::~LibraryLoader() {
  if (autoUnload) {
    for (map<string, boost::shared_ptr<LibraryData> >::iterator iter = libraryTable.begin();
        iter != libraryTable.end(); ++iter)
    {
      iter->second->unload();
    }
  }
}

LibraryLoader& LibraryLoader::getInstance() {
  if (!instance)
    instance = new LibraryLoader();
  return *instance;
}

void LibraryLoader::setAutoUnload(bool autoUnload) {
  this->autoUnload = autoUnload;
}

bool LibraryLoader::getAutoUnload() const {
  return autoUnload;
}

void LibraryLoader::loadLibrary(const string& filename) {
  // If loaded, increase counter, else load
  map<string, boost::shared_ptr<LibraryData> >::iterator iter = libraryTable.find(filename);
  if (iter == libraryTable.end()) {
    boost::shared_ptr<LibraryData> data(new LibraryData(filename.c_str()));
    libraryTable[filename] = data;
//    cout << filename << " library loaded." << endl;
  } else {
    iter->second->loadCount = iter->second->loadCount + 1;
    //cout << filename << " library counter incremented (" << iter->second->loadCount << ")." << endl;
  }
}

void LibraryLoader::unloadLibrary(const string& filename) {
//  cout << "Try to free library " << filename << endl;
  map<string, boost::shared_ptr<LibraryData> >::iterator iter = libraryTable.find(filename);
  if (iter != libraryTable.end()) {
    boost::shared_ptr<LibraryData> data = iter->second;
    data->loadCount = data->loadCount - 1;
//    cout << filename << " library counter decremented (" << data->loadCount << ")." << endl;
    if (!data->loadCount) {
      data->unload();
      libraryTable.erase(filename);
//      cout << "Library freed." << endl;
    }
  }
}

bool LibraryLoader::librariesUpdated() {
  bool updated = false;
  time_t lastModified = 0;
  for (map<string, boost::shared_ptr<LibraryData> >::iterator iter = libraryTable.begin();
      iter != libraryTable.end(); ++iter)
  {
    boost::shared_ptr<LibraryData> data = iter->second;
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
  for (map<string, boost::shared_ptr<LibraryData> >::iterator iter = libraryTable.begin();
        iter != libraryTable.end(); ++iter)
  {
    boost::shared_ptr<LibraryData> data = iter->second;
    for (size_t i = 0; i < data->classnames.size(); ++i) {
      if (data->classnames[i].compare(classname) == 0)
        return iter->first;
    }
  }

  return "";
}

}
