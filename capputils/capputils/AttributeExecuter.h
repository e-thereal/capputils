/**
 * \brief Contains the \c AttributeExecuter class
 * \file AttributeExecuter.h
 *
 * \date Mar 2, 2011
 * \author Tom Brosch
 */

#ifndef CAPPUTILS_ATTRIBUTEEXECUTER_H_
#define CAPPUTILS_ATTRIBUTEEXECUTER_H_

#include <capputils/capputils.h>
#include <capputils/reflection/ReflectableClass.h>
#include <capputils/reflection/IClassProperty.h>

namespace capputils {

/**
 * \brief Used internally in conjunction with the \c ExecutableAttribute
 *
 * This class is used to invoke \c executeBefore() and \c executeAfter() methods before and after
 * a property has been set using the appropriate setter method.
 */
class CAPPUTILS_API AttributeExecuter {
public:

  /**
   * \brief Invokes the \c executeBefore() method of \c ExecutableAttributes
   *
   * \param[in] object    Object having the property \a property
   * \param[in] property  Property whose executable attributes are about to be executed.
   */
  static void ExecuteBefore(reflection::ReflectableClass& object,
      const reflection::IClassProperty& property);

  /**
     * \brief Invokes the \c executeAfter() method of \c ExecutableAttributes
     *
     * \param[in] object    Object having the property \a property
     * \param[in] property  Property whose executable attributes are about to be executed.
     */
  static void ExecuteAfter(reflection::ReflectableClass& object,
        const reflection::IClassProperty& property);
};

}

#endif /* CAPPUTILS_ATTRIBUTEEXECUTER_H_ */
