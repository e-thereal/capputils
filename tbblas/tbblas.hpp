#ifndef TBBLAS_HPP
#define TBBLAS_HPP

#include <iostream>
#include <thrust/version.h>

#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA
#error "Is is assumed that CUDA is used as the device backend. For parallelized CPU code, change the host backend instead."
#endif

#if THRUST_VERSION < 100600
#error "At least thrust version 1.6 is required."
#endif

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

#if defined(i386) && !defined(TBBLAS_DISABLE_CBLAS)
#define TBBLAS_HAVE_CBLAS
#endif

#define TBBLAS_HAVE_CUBLAS

#endif
