#ifndef TBBLAS_DISK_TENSOR_HPP_
#define TBBLAS_DISK_TENSOR_HPP_

#include "tbblas.hpp"
#include "tensor_base.hpp"

namespace tbblas {

/**

 Supports the following operations:

 device_tensor_t dt;

 disk_tensor_t disk = tbblas::copy(dt);
 
 and

 dt = tbblas::copy(disk);

 plus:

 disk.size();


 What else:
 - I need to store the content somewhere temporary, so I need to save the file name.
 - I have function that creates a boost::shared_ptr<host_tensor> containing the content
 - need to overload operator=(const tbblas_tensor_copy& op)

 */

template<class T, unsigned dim>
class disk_tensor {

};

}

#endif /* TBBLAS_DISK_TENSOR_HPP_ */