#ifndef GPUSVM_SVMCOMMONH
#define GPUSVM_SVMCOMMONH

#include <string>

namespace gpusvm {

struct Kernel_params{
	float gamma;
	float coef0;
	int degree;
	float b;
	std::string kernel_type;
};

enum SelectionHeuristic {FIRSTORDER, SECONDORDER, RANDOM, ADAPTIVE};

}

#endif
