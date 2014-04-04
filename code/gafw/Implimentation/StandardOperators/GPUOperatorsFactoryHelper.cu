/* GPUOperatorsFactoryHelper.cu:  implementation of the GPUOperatorsFactoryHelper class 
 *      
 * Copyright (C) 2013  Daniel Muscat
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 * Author's contact details can be found at http://www.danielmuscat.com 
 *
 */
#include <GPUafw.h>
#include "GPUOperatorsFactoryHelper.h"
#include "Equals.h"
#include "cuFFTException.h"

#include "MultiFFT2D.h"
#include "ZeroToCentre2DShift.h"
#include "ZeroToCorner2DShift.h"
#include "ArraysToVector.h"
#include "IFFT2D.h"
#include "AccumulatedSum.h"
#include "ValueChangeDetect.h"
#include "ZeroPad.h"
#include "Zeros.h"
#include "Real.h"
#include "Complex.h"
#include "SingleMultiply.h"


#include "OperatorsFromTemplates/AbsMax.h"
#include "OperatorsFromTemplates/Chi2.h"
#include "OperatorsFromTemplates/GriddedPointsCalculator.h"
#include "OperatorsFromTemplates/MaxChi2.h"
#include "OperatorsFromTemplates/MaxMetric.h"
#include "OperatorsFromTemplates/RootSumSquare.h"
#include "OperatorsFromTemplates/Sum.h"
#include "OperatorsFromTemplates/SumDifference.h"
#include "OperatorsFromTemplates/SumMultiply.h"

#include <string>

using namespace GPUAFW::StandardOperators;
using namespace GAFW::GPU::StandardOperators;

GPUOperatorsFactoryHelper::GPUOperatorsFactoryHelper():Identity("GPUStandardOperators","GPUStandardOperators")
{
    this->isGPUFactory=false;
    this->factory=NULL;
}


GPUOperatorsFactoryHelper::~GPUOperatorsFactoryHelper()
{
    
}



GAFW::FactoryHelper* GPUOperatorsFactoryHelper::reCreateForFactory(GAFW::Factory *f)
{
    GPUOperatorsFactoryHelper *ret=new GPUOperatorsFactoryHelper();
    if (f->getName()==string("GPUFactory"))
        ret->isGPUFactory=true;
    else
        ret->isGPUFactory=false;
    ret->factory=dynamic_cast<GAFW::GPU::GPUFactory *> (f);
    return ret;
    
}

GAFW::ArrayOperator *GPUOperatorsFactoryHelper::createOperator(string nickname, string name)
{
    if (!this->isGPUFactory) return NULL;  
        //This helper works only for the GPU factory so if not helping such a factory just return NULL

    if (name=="equals")
        return new Equals(this->factory,nickname);
    if (name=="IFFT2D")
        return new IFFT2D(this->factory,nickname);
    if (name=="multiFFT2D")
        return new MultiFFT2D(this->factory,nickname);
    if (name=="ZeroToCentre2DShift")
        return new ZeroToCentre2DShift(this->factory,nickname);
    if (name=="ZeroToCorner2DShift")
        return new ZeroToCorner2DShift(this->factory,nickname);
    if (name=="Zeros")
        return new Zeros(this->factory,nickname);
    if (name=="Real")
        return new Real(this->factory,nickname);
    if (name=="Complex")
        return new Complex(this->factory,nickname);
    if (name=="SingleMultiply")
        return new SingleMultiply(this->factory,nickname);
    if (name=="ArraysToVector")
        return new ArraysToVector(this->factory,nickname);
    if (name=="AccumulatedSum")
        return new AccumulatedSum(this->factory,nickname);
    if (name=="Chi2")
        return new Chi2(this->factory,nickname);
    if (name=="ValueChangeDetect")
        return new ValueChangeDetect(this->factory,nickname);
    if (name=="ZeroPad")
        return new ZeroPad(this->factory,nickname);
    if (name=="Chi2new")
        return new Chi2(this->factory,nickname);
    if (name=="RootSumSquare")
        return new RootSumSquare(this->factory,nickname);
    if (name=="MaxMetric")
        return new MaxMetric(this->factory,nickname);
    if (name=="MaxChi2")
        return new MaxChi2(this->factory,nickname);
    if (name=="SumDifference")
        return new SumDifference(this->factory,nickname);
    if (name=="SumMultiply")
        return new SumMultiply(this->factory,nickname);
    if (name=="AbsMax")
        return new AbsMax(this->factory,nickname);
    if (name=="Sum")
        return new Sum(this->factory,nickname);
    if (name=="GriddedPointsCalculator")
        return new GriddedPointsCalculator(this->factory,nickname);
    return NULL;
}

   
