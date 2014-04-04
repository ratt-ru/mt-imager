/* PreIFFTUVCorrection.cpp:  Pure C++ implementation of the PreIFFTUVCorrection operator 
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
PreIFFTUVCorrection::PreIFFTUVCorrection(GPUFactory * factory,std::string nickname) : GPUArrayOperator(factory,nickname,string("Operator::PreIFFTUVCorrection"))
{
    
}
PreIFFTUVCorrection::~PreIFFTUVCorrection()
{

}

void PreIFFTUVCorrection::validate(){
    /********************************************************************
     * VAlidations:                                                     *
     * 1. 1 input Matrix which must have valid dimensions
     * 1. Only one inputs and one output is supported
     * 
     * Output matrix will be set to same dimensions and store type
     * of input matrix
     * ******************************************************************/
  
   //No of inputs
    if (this->inputs.size()!=1) 
    {
        throw ValidationException("Only one input is supported");
    }
    
    if (this->outputs.size()!=1) throw ValidationException("Only one output is supported");
    
    if (!this->inputs[0]->isDefined()) throw ValidationException("Input is not defined");
    if (this->inputs[0]->getDimensions().getNoOfDimensions()!=3) throw ValidationException("Expected a 3D Array (lengthxwidthxpolarization");
    //Seems all input is as it should be
    
    ArrayDimensions d=this->inputs[0]->getDimensions();
    ArrayDimensions newDim(3,d.getX(),d.getZ(),d.getY());
    this->outputs[0]->setDimensions(newDim);
    this->outputs[0]->setType(this->inputs[0]->getType());
}
