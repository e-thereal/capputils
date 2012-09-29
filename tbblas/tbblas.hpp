#ifndef TBBLAS_HPP
#define TBBLAS_HPP

#ifdef _WIN32
#ifndef TBBLAS_EXPORTS
#pragma comment(lib, "tbblas")
#endif
#endif

#ifndef TBBLAS_UNUSED
#define TBBLAS_UNUSED(a) (void)a
#endif

#endif
