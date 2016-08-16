#ifndef TBBLAS_HPP
#define TBBLAS_HPP

#include <iostream>
#include <thrust/version.h>
#include <csignal>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

//#define TBBLAS_ALLOC_WARNING_ENABLED

#if defined(TBBLAS_INTERRUPT_ALLOC_ENABLED)
#define TBBLAS_ALLOC_WARNING std::cerr << "[Warning] Allocating memory at " << __FILE__ << ":" << __LINE__ << std::endl; raise(SIGINT);
#define TBBLAS_FREE_MESSAGE std::cerr << "[Message] Freeing memory at " << __FILE__ << ":" << __LINE__ << std::endl; raise(SIGINT);
#elif defined(TBBLAS_ALLOC_WARNING_ENABLED)
#define TBBLAS_ALLOC_WARNING std::cerr << "[Warning] Allocating memory at " << __FILE__ << ":" << __LINE__ << std::endl;
#define TBBLAS_FREE_MESSAGE std::cerr << "[Message] Freeing memory at " << __FILE__ << ":" << __LINE__ << std::endl;
#else
#define TBBLAS_ALLOC_WARNING
#define TBBLAS_FREE_MESSAGE
#endif

#if defined(i386) && !defined(TBBLAS_DISABLE_CBLAS)
#define TBBLAS_HAVE_CBLAS
#endif

// I need a better way to configure tbblas

//#define TBBLAS_HAVE_CUBLAS
#define TBBLAS_CPU_ONLY
#define TBBLAS_CNN_NO_SELFTEST
#define TBBLAS_CONV_RBM_NO_SELFTEST

#endif
