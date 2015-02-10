/*
 * io.hpp
 *
 *  Created on: 2014-10-05
 *      Author: tombr
 */

#ifndef TBBLAS_IMGPROC_IO_HPP_
#define TBBLAS_IMGPROC_IO_HPP_

#include <iostream>
#include <tbblas/imgproc/fmatrix4.hpp>

namespace tbblas {

namespace ip {

std::ostream& operator<<(std::ostream& os, const fmatrix4& mat) {
  const int precision = 5;

  os << "[4 x 4]" << std::endl;
  os << std::setprecision(precision) << std::setw(precision + 6) << mat.r1.x << " ";
  os << std::setprecision(precision) << std::setw(precision + 6) << mat.r1.y << " ";
  os << std::setprecision(precision) << std::setw(precision + 6) << mat.r1.z << " ";
  os << std::setprecision(precision) << std::setw(precision + 6) << mat.r1.w << " ";
  os << std::endl;
  os << std::setprecision(precision) << std::setw(precision + 6) << mat.r2.x << " ";
  os << std::setprecision(precision) << std::setw(precision + 6) << mat.r2.y << " ";
  os << std::setprecision(precision) << std::setw(precision + 6) << mat.r2.z << " ";
  os << std::setprecision(precision) << std::setw(precision + 6) << mat.r2.w << " ";
  os << std::endl;
  os << std::setprecision(precision) << std::setw(precision + 6) << mat.r3.x << " ";
  os << std::setprecision(precision) << std::setw(precision + 6) << mat.r3.y << " ";
  os << std::setprecision(precision) << std::setw(precision + 6) << mat.r3.z << " ";
  os << std::setprecision(precision) << std::setw(precision + 6) << mat.r3.w << " ";
  os << std::endl;
  os << std::setprecision(precision) << std::setw(precision + 6) << mat.r4.x << " ";
  os << std::setprecision(precision) << std::setw(precision + 6) << mat.r4.y << " ";
  os << std::setprecision(precision) << std::setw(precision + 6) << mat.r4.z << " ";
  os << std::setprecision(precision) << std::setw(precision + 6) << mat.r4.w << " ";
  os << std::endl;
  return os;
}

}

}

#endif /* TBBLAS_IMGPROC_IO_HPP_ */
