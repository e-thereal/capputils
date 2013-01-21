#include <boost/math/special_functions/binomial.hpp>

namespace tbblas {

double binomial(int n, int k) {
  return boost::math::binomial_coefficient<double>(n, k);
}

}
