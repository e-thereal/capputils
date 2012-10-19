#ifndef GPUSVM_CACHEH
#define GPUSVM_CACHEH

#include <vector>
#include <list>

namespace gpusvm {

class Cache {
 public:
  Cache(int nPointsIn, int cacheSizeIn);
  ~Cache();
  void findData(const int index, int &offset, bool &compute);
	void search(const int index, int &offset, bool &compute);
  void printCache();
	void printStatistics();

public:
  int nPoints;
  int cacheSize;
  class DirectoryEntry {
  public:
    enum {NEVER, EVICTED, INCACHE};
    DirectoryEntry();
    int status;
    int location;
    std::list<int>::iterator lruListEntry;
  };

  std::vector<DirectoryEntry> directory;
  std::list<int> lruList;
  int occupancy;
  int hits;
  int compulsoryMisses;
  int capacityMisses;
};

}

#endif
