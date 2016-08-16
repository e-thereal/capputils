#pragma once

#ifndef _CAPPUTILS_VOLATILE_H_
#define _CAPPUTILS_VOLATILE_H_

#include <capputils/attributes/IAttribute.h>

namespace capputils {

namespace attributes {

class VolatileAttribute : public virtual IAttribute {
};

AttributeWrapper* Volatile();

}

}

#endif
