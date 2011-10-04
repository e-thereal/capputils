/**
 * \brief Contains the flag attribute
 * \file Flag.h
 *
 * \date Feb 10, 2011
 * \author Tom Brosch
 */

#ifndef CAPPUTILS_FLAG_H_
#define CAPPUTILS_FLAG_H_

#include "IAttribute.h"

namespace capputils {

namespace attributes {

/**
 * \brief Makes a class property a boolean flag which is either set or not.
 *
 * This attribute changes the parsing behavior of the \c ArgumentParser. The according
 * parameter switch does not require any succeeding arguments. If the switch is detected,
 * the according property is set to true.
 */
class FlagAttribute: public virtual IAttribute {
public:
  /**
   * \brief Constructs a new \c FlagAttribute
   *
   * Use the Flag() function which returns a wrapped FlagAttribute when passing
   * this attribute to an argument list.
   */
  FlagAttribute();

  /**
   * \brief Destructs the flag attribute
   */
  virtual ~FlagAttribute();
};

/**
 * \brief Returns a wrapped \c FlagAttribute
 */
AttributeWrapper* Flag();

}

}

#endif /* CAPPUTILS_FLAG_H_ */
