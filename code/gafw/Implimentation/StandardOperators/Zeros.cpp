/* Zeros.cpp:  C++ implementation of the Zeros operator 
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
#include "Zeros.h"

#include <string>
using namespace GAFW::GPU::StandardOperators;
using namespace GAFW::GeneralImplimentation;

Zeros::Zeros(GAFW::GPU::GPUFactory* factory,string nickname):GPUArrayOperator(factory,"Zeros")
{
    
}

Zeros::~Zeros() {
}
void Zeros::validate(){
    /********************************************************************
     * VAlidations:                                                     *
     * 1. One or zero input Matrixes *
     * 2. If no input matrix Output matrix should be defined (valid store
     *   type and dimensions           *
     * 3.) if input matrix exist output matrix will be defined as per input matrix
     * 
     * Output matrix will be set to same dimensions 
     * of input matrixes 
     * 
     * Output Matrix will be set comples_float if any of the input is complex
     * it will be set to real_float otherwise
     * ******************************************************************/
  
   //No of inputs
    if ((this->inputs.size()!=0)&&(this->inputs.size()!=1)) throw ValidationException("Either 1 or 0 inputs are expected for this operator");
    if (this->outputs.size()!=1) throw ValidationException("Only one output is expected");
    
    //When no inpit exists
    if (this->inputs.size()==0)
    {
        if (!this->outputs[0]->isDefined()) throw ValidationException ("Output is expected to be defined when no input is set");
    }
    else
    {
        if (!this->inputs[0]->isDefined()) throw ValidationException ("Input is expected to be defined when given");
        ArrayDimensions arr=this->inputs[0]->getDimensions();
        this->outputs[0]->setDimensions(arr);
        this->outputs[0]->setType(this->inputs[0]->getType());
        
    }
    logDebug(validation,"Validation successful");

}

    
