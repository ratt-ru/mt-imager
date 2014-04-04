/* ZeroToCorner2DShift.cpp:  C++ implementation of the ZeroToCorner2DShift operator 
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
#include "ZeroToCorner2DShift.h"
using namespace GAFW::GPU::StandardOperators;
using namespace GAFW::GeneralImplimentation;

ZeroToCorner2DShift::ZeroToCorner2DShift(GAFW::GPU::GPUFactory * factory,std::string nickname):GPUArrayOperator(factory,nickname,string("Operator::FFT2DShift"))
{


}
ZeroToCorner2DShift::~ZeroToCorner2DShift() {
}

void ZeroToCorner2DShift::validate()
{
    
  
   //No of inputs
    if (this->inputs.size()!=1) throw ValidationException("Only 1 input is supported");
    if (this->outputs.size()!=1) throw ValidationException("Only 1 output is supported");
    //Ok they seem to be equal but have to make sure that infact they have been set
    if (!this->inputs[0]->isDefined()) throw ValidationException("Input is not defined");
    
    switch (this->inputs[0]->getType())
    {
        case complex_float:
        case complex_double:
            //OK
            break;
        case real_float:
            throw ValidationException("At current FFT2DShift of a real array is not supported");
        default:
            throw ValidationException("Unsupported store type of input");
            
    }
    
    //We expect input to be a 2D Matrix
    
    ArrayDimensions d=this->inputs[0]->getDimensions();
    if (d.getNoOfDimensions()<2) throw ValidationException("FFT2DShift can work on arrays of 2 dimensions or above!")
    this->outputs[0]->setDimensions(d);
    this->outputs[0]->setType(this->inputs[0]->getType());
}
