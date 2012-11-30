#ifndef TBBLAS_HPP
#define TBBLAS_HPP

#include <iostream>

#ifdef _WIN32
#ifndef TBBLAS_EXPORTS
#pragma comment(lib, "tbblas")
#endif
#endif

#ifndef TBBLAS_UNUSED
#define TBBLAS_UNUSED(a) (void)a
#endif

#ifdef TBBLAS_ALLOC_WARNING_ENABLED
#define TBBLAS_ALLOC_WARNING std::cerr << "[Warning] Allocating memory at " << __FILE__ << ":" << __LINE__ << std::endl;
#else
#define TBBLAS_ALLOC_WARNING
#endif

#endif
