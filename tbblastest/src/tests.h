/*
 * tests.h
 *
 *  Created on: Nov 4, 2011
 *      Author: tombr
 */

#ifndef TESTS_H_
#define TESTS_H_

void helloworld();
void convtest();
void sumtest();
void entropytest();
void proxycopy();
void ffttest();
void fftbenchmarks();
void convtest2();
void scalarexpressions();
void proxytests();
void convrbmtests();
void partialffttest();
void fftflip();
void maskstest();
void multigpu();
void copytest();
void ompsegfault();
void benchmarks();
void rearrangetest();
void trainertests(int filterCount, int channelCount, int reps, int convnetReps);
void fasttrainer(int size, int channelCount, int filterCount, int reps);
void synctest();

#endif /* TESTS_H_ */
