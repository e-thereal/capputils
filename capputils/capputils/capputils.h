#pragma once
#ifndef _CAPPUTILS_CAPPUTILS_H_
#define _CAPPUTILS_CAPPUTILS_H_

#ifdef _WIN32
#ifdef CAPPUTILS_EXPORTS
//#define CAPPUTILS_API __declspec(dllexport)
#else
#pragma comment(lib, "capputils")
//#define CAPPUTILS_API __declspec(dllimport)
#endif
#else
//#define CAPPUTILS_API
#endif

#define CAPPUTILS_API

#endif