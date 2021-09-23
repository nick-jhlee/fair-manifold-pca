/*
This is the test file to run the problem defined in LRBlindDeconvolution.h and LRBlindDeconvolution.cpp.

---- WH
*/

#ifndef TESTCFRANKQ2FBLINDDECON2D_H
#define TESTCFRANKQ2FBLINDDECON2D_H


#include <iostream>
#include "Others/randgen.h"
#include "Manifolds/Manifold.h"
#include "Problems/Problem.h"
#include <ctime>

#include "test/DriverMexProb.h"

#include "Problems/CFRankQ2FBlindDecon2D.h"
#include "Manifolds/SphereTx.h"
#include "Manifolds/CFixedRankQ2F.h"

#include "Solvers/RSD.h"
#include "Solvers/RNewton.h"
#include "Solvers/RCG.h"
#include "Solvers/RBroydenFamily.h"
#include "Solvers/RWRBFGS.h"
#include "Solvers/RBFGS.h"
#include "Solvers/LRBFGS.h"

#include "Solvers/RTRSD.h"
#include "Solvers/RTRNewton.h"
#include "Solvers/RTRSR1.h"
#include "Solvers/LRTRSR1.h"

#include "Others/def.h"

#include "Others/fftw/fftw3.h"
#undef abs

using namespace ROPTLIB;

#ifdef ROPTLIB_WITH_FFTW

void testCFRankQ2FBlindDecon2D(void);

#endif
#endif
