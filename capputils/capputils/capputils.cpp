// capputils.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "capputils.h"


// This is an example of an exported variable
CAPPUTILS_API int ncapputils=0;

// This is an example of an exported function.
CAPPUTILS_API int fncapputils(void)
{
	return 42;
}

// This is the constructor of a class that has been exported.
// see capputils.h for the class definition
Ccapputils::Ccapputils()
{
	return;
}
