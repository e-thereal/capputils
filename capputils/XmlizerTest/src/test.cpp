/*
 * test.cpp
 *
 *  Created on: Jul 16, 2012
 *      Author: tombr
 */
#include <memory>

#include <boost/shared_ptr.hpp>

class A {
public:
  virtual ~A() { }
};

class B : public A { };

void test() {
  std::shared_ptr<A> ptr = std::make_shared<B>();
  std::shared_ptr<B> ptr2 = std::dynamic_pointer_cast<B>(ptr);
}


