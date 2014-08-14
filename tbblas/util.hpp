/*
 * util.hpp
 *
 *  Created on: Apr 7, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_UTIL_HPP_
#define TBBLAS_UTIL_HPP_

namespace tbblas {

extern int peer_access_enabled_count;
void enable_peer_access(int gpu_count);
void disable_peer_access(int gpu_count);

void synchronize();

}


#endif /* TBBLAS_UTIL_HPP_ */
