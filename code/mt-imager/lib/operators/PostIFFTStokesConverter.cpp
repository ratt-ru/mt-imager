/* PostIFFTStokesConverter.cpp:  Pure C++ implementation of the PostIFFTStokesConverter operator 
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

#include "cuda_runtime.h"
#include "ImagerOperators.h"
#include <sstream>
using namespace mtimager;
using namespace GAFW;
using namespace GAFW::GPU;
using namespace GAFW::GeneralImplimentation;
using namespace GAFW::Tools::CppProperties;
PostIFFTStokesConverter::PostIFFTStokesConverter(GPUFactory * factory,std::string nickname): GPUArrayOperator(factory,nickname,string("Operator::PostIFFTStokesConverter"))
{
    
}
PostIFFTStokesConverter::~PostIFFTStokesConverter()
{

}

void PostIFFTStokesConverter::validate(GAFW::GPU::ValidationData &data){
    
  
   //No of inputs
    if (this->inputs.size()!=2) 
    {
        throw ValidationException("Two inputs are expected");
    }
    this->checkParameter("PolarizationType",Properties::Int);
    if ((this->getIntParameter("PolarizationType")!=0) && (this->getIntParameter("PolarizationType")!=1))
          throw ValidationException("PolarizationType parameter is expected to be 0 (linear Polarisation) or 1(Circular Polarisation)");
    
    if (this->outputs.size()!=1) throw ValidationException("Only one output is supported");
    
    if (this->inputs[0]->getDimensions().getNoOfDimensions()!=3) throw ValidationException("Expected a 3D Array (polarizationxlengthxwidth)");
    
    if (this->inputs[0]->getDimensions().getZ()!=this->inputs[1]->getDimensions().getTotalNoOfElements())
    {
        std::stringstream ss;
        ss<<"The No Of polarizations in imager and normalizer are not the same. Total elements is normalizer array is "<< this->inputs[1]->getDimensions().getTotalNoOfElements();
        throw ValidationException(ss.str());
    }
    
    
    if (this->inputs[1]->getType()!=real_float) throw ValidationException("Normaliser input type expected to be real_float");
    
    //Seems all input is as it should be
    ArrayDimensions d=this->inputs[0]->getDimensions();
    this->outputs[0]->setDimensions(d);
    this->outputs[0]->setType(real_float);
}

