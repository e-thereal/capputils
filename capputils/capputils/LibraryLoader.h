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

//#include <boost/signals.hpp>
#include <boost/shared_ptr.hpp>

#include <capputils/EventHandler.h>

namespace capputils {

#ifdef _WIN32
struct HandleWrapper;
#endif

struct LibraryData {
  std::string filename;
  time_t lastModified;
  int loadCount;
  std::vector<std::string> classnames; ///< contains all classes that come with the library
#ifdef _WIN32
  HandleWrapper* handleWrapper;
#else
  void* handle;
#endif
  LibraryData(const char* filename);
  void unload();
  virtual ~LibraryData();
};

class LibraryLoader {
private:
  static LibraryLoader* instance;
  std::map<std::string, boost::shared_ptr<LibraryData> > libraryTable;
  bool autoUnload;

protected:
  LibraryLoader();

public:
  virtual ~LibraryLoader();

  static LibraryLoader& getInstance();

  void setAutoUnload(bool autoUnload);
  bool getAutoUnload() const;

  void loadLibrary(const std::string& filename);
  void unloadLibrary(const std::string& filename);
  bool librariesUpdated();

  // Returns the library name that defines the given class
  std::string classDefinedIn(const std::string& classname);
};

}

#endif /* LIBRARYLOADER_H_ */
