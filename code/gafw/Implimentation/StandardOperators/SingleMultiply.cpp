/* SingleMultipy.cpp:  C++ implementation of the SingleMultiply operator 
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


#include "GPUafw.h"
#include "cuda_runtime.h"
#include "SingleMultiply.h"
#include <sstream>
using namespace GAFW::GPU::StandardOperators;
using namespace GAFW::Tools::CppProperties;
using namespace GAFW::GeneralImplimentation;

SingleMultiply::SingleMultiply(GAFW::GPU::GPUFactory * factory,std::string nickname) :GPUArrayOperator(factory,nickname,string("Operator::SingleMultiply"))
{
    
}
SingleMultiply::~SingleMultiply()
{

}

void SingleMultiply::validate(){
    /********************************************************************
     * VAlidations:                                                     *
     * 1. 1 input Matrix which must have valid dimensions
     * 1. Only one inputs and one output is supported
     * 
     * Output matrix will be set to same dimensions and store type
     * of input matrix
     * ******************************************************************/
  //SingleMultiply can work in two modes 
   //No of inputs
    
    if ((this->inputs.size()!=1)&&(this->inputs.size()!=2)) 
    {
        std::stringstream s;
        s << "Only one or two inputs are supported. " << this->inputs.size() << " inputs found.";
        throw ValidationException(string(s.str()));
        
    }
    
    if (this->outputs.size()!=1) throw ValidationException("Only one output is supported");
    
    if (!this->inputs[0]->isDefined()) throw ValidationException("Input is not defined"); //this is the input array.. it can be real or complex
    //Now we need to check about the multiplier
    Properties::PropertyType type;

    if (this->inputs.size()==1)
    {
        //Ok the multiplier value have to be stored in the parameter list
        //we check param multiplier.value
        if (!this->params.isPropertySet("multiplier.value")) throw ValidationException("Parameter multiplier.value is not set and one input given");
        type=this->params.getPropertyType("multiplier.value");
        if ((type!=Properties::Complex)&&(type!=Properties::Float))
            throw ValidationException("Parameter multiplier.value set to a wrong type");
    }
    if (this->inputs.size()==2)
    {
        //Ok multiplier is stored in matrix... but it must contain only one element
        if (this->inputs[1]->getDimensions().getTotalNoOfElements()!=1)
            throw ValidationException("input 2 has more then one element");
        switch (this->inputs[1]->getType())
        {
            case complex_float:
                type=Properties::Complex;
                break;
            case real_float:
                type=Properties::Float;
                break;
            case real_double:
                type=Properties::Double;
                break;
              
            default:
                throw ValidationException("2nd input is not of the expected type (either complex-float or real_float)");
        }
    }
    //Ok now we have all data and seems all input is good
    
    ArrayDimensions d=this->inputs[0]->getDimensions();
    this->outputs[0]->setDimensions(d);
    StoreType storeType=this->inputs[0]->getType();
    if (type==Properties::Complex) storeType=complex_float;
    this->outputs[0]->setType(storeType);
    //that's it
}
