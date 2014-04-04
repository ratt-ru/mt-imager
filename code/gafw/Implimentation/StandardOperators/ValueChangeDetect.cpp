/* ValueChangeDetect.cpp:  C++ implementation of the ValueChangeDetect operator 
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
#include "GPUafw.h"
#include "ValueChangeDetect.h"
#include <sstream>
using namespace GAFW::GPU::StandardOperators;
using namespace GAFW::GeneralImplimentation;


ValueChangeDetect::ValueChangeDetect(GAFW::GPU::GPUFactory * factory,std::string nickname) : GPUArrayOperator(factory,nickname,string("Operator::ValueChangeDetect"))
{
    
}
ValueChangeDetect::~ValueChangeDetect()
{

}

void ValueChangeDetect::validate(){
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
        std::stringstream s;
        s << "Only one input is supported. " << this->inputs.size() << " inputs found.";
        throw ValidationException(string(s.str()));
        
    }
    ArrayDimensions d=this->inputs[0]->getDimensions();
    if (d.getNoOfDimensions()!=1) throw ValidationException("Only one dimensional arrays are supported")
    if ((this->outputs.size()!=1)) throw ValidationException("Expected only one input");
    //Seems all input is as it should be
    
    this->outputs[0]->setDimensions(d);
    this->outputs[0]->setType(real_int);
    
}
