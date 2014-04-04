/* ConvFunctionReOrganize.cpp:  Pure C++ implementation of the ConvFunctionReOrganize operator 
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
ConvFunctionReOrganize::ConvFunctionReOrganize(GPUFactory * factory,std::string nickname) :GPUArrayOperator(factory,nickname,string("Operator::ConvFunctionReOrganize"))
{
    
}
ConvFunctionReOrganize::~ConvFunctionReOrganize()
{

}

void ConvFunctionReOrganize::validate(){
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
        throw ValidationException("Only one input is expected");
    }
    
    if (this->outputs.size()!=1) throw ValidationException("Only one output is supported");
    
    if (!this->inputs[0]->isDefined()) throw ValidationException("Input 0 is not defined");
    
    if (this->inputs[0]->getDimensions().getNoOfDimensions()!=2) throw ValidationException("Expected a 2D Array");
    //Seems all input is as it should be
    if(!this->isParameterSet("ConvolutionFunction.sampling")) throw ValidationException("Parameter ConvolutionFunction.sampling is not set");
    if(!this->isParameterSet("ConvolutionFunction.support")) throw ValidationException("Parameter ConvolutionFunction.support is not set");
    int sampling=this->getIntParameter("ConvolutionFunction.sampling");
    int support=this->getIntParameter("ConvolutionFunction.support");
    if (support<1) throw ValidationException("Support is less then 1!!"); 
    
    ArrayDimensions d=this->inputs[0]->getDimensions();
   
    if (d.getNoOfColumns()!=d.getNoOfRows()) throw ValidationException("Convolution Functions are expected to be square");
    if (sampling%2)
    {
        if ((sampling*support)>d.getNoOfColumns()) throw ValidationException("Size of input convolution functions is smaller support*sampling");
    }
    else
    {
        if ((sampling*support+1)>d.getNoOfColumns()) throw ValidationException("Size of input convolution functions is smaller then support*sampling+1");
    }
        
            
    //There is a slightly different configuration for even sampling we need extra space
    ArrayDimensions dout(1);
    if (sampling%2)
    {
        dout.setX(sampling*support*sampling*support);
    }
    else
    {
        dout.setX((sampling+1)*support*(sampling+1)*support);
    }
    this->outputs[0]->setDimensions(dout);
    this->outputs[0]->setType(this->inputs[0]->getType()); //in truly we still have to restrict

}

