/* ConvFunctionSupportFind.cpp:  Pure C++ implementation of the ConvFunctionSupportFind operator 
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



#include "ImagerOperators.h"
#include <sstream>
#include <iostream>
using namespace mtimager;
using namespace GAFW;
using namespace GAFW::GPU; using namespace GAFW::GeneralImplimentation;

ConvFunctionSupportFind::ConvFunctionSupportFind(GPUFactory * factory,std::string nickname):GPUArrayOperator(factory,nickname,string("Operator::ConvFunctionReOrganize"))
{
    
}
ConvFunctionSupportFind::~ConvFunctionSupportFind()
{

}

void ConvFunctionSupportFind::validate(){
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
    if ((this->inputs[0]->getDimensions().getNoOfDimensions()!=3)&&(this->inputs[0]->getDimensions().getNoOfDimensions()!=2)) throw ValidationException("Expected a 3D/2D Array");
    //Seems all input is as it should be
    if(!this->isParameterSet("ConvolutionFunction.sampling")) throw ValidationException("Parameter ConvolutionFunction.sampling is not set");
    if (!this->isParameterSet("ConvolutionFunction.takeaszero")) throw ValidationException("Parameter ConvolutionFunction.takeaszero is not set");
    //We should also check if float

    //check that dimensions do correspond to sampling and support
    if (this->inputs[0]->getDimensions().getNoOfDimensions()==2)
        this->outputs[0]->setDimensions(ArrayDimensions(1,1));
    else
        this->outputs[0]->setDimensions(ArrayDimensions(1,this->inputs[0]->getDimensions().getZ()));
    this->outputs[0]->setType(real_float);  //should be integer but not yet supported 
}

