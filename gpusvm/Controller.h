#ifndef GPUSVM_CONTROLLER_H_
#define GPUSVM_CONTROLLER_H_

#include <vector>
#include <sys/time.h>
#include "svmCommon.h"
#include <cstdlib>

namespace gpusvm {

class Controller {
 public:
  Controller(float initialGap, SelectionHeuristic currentMethodIn, int samplingIntervalIn, int problemSize);
  void addIteration(float gap);
  void print();
  SelectionHeuristic getMethod();
 private:
  bool adaptive;
  int samplingInterval;
  std::vector<float> progress;
  std::vector<int> method;
  SelectionHeuristic currentMethod;
  std::vector<float> rates;
  int timeSinceInspection;
  int inspectionPeriod;
  int beginningOfEpoch;
  int middleOfEpoch;
  int currentInspectionPhase;
  float filter(int begin, int end);
  float findRate(struct timeval* start, struct timeval* finish, int beginning, int end);
  struct timeval start;
  struct timeval mid;
  struct timeval finish;
};

}

#endif /* GPUSVM_CONTROLLER_H_ */
