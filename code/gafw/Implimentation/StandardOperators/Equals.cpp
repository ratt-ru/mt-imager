/* Equals.cpp:  Pure C++ implementation of the Equals operator 
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
#include "cuComplex.h"
#include "GPUafw.h"
#include "Equals.h"
#include <sstream>
using namespace GAFW::GPU::StandardOperators;
using namespace GAFW::GeneralImplimentation;

Equals::Equals(GAFW::GPU::GPUFactory * factory,std::string nickname) :GPUArrayOperator(factory,nickname,string("Operator::Equals"))
{
    
}
Equals::~Equals()
{

}

void Equals::validate(){
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
    
    if (this->outputs.size()!=1) throw ValidationException("Only one output is supported");
    
    if (!this->inputs[0]->isDefined()) throw ValidationException("Input is not defined");
    
    //Seems all input is as it should be
    
    ArrayDimensions d=this->inputs[0]->getDimensions();
    this->outputs[0]->setDimensions(d);
    this->outputs[0]->setType(this->inputs[0]->getType());
    
}
void Equals::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
    if (data.inputs[0].pointer==data.outputs[0].pointer) return;  //Nothing to do
    else
    {
        cudaError_t e=cudaMemcpyAsync(data.outputs[0].pointer,
                data.inputs[0].pointer,
                this->calculateArraySize(data.inputs[0].dim,data.inputs[0].type),
                cudaMemcpyDeviceToDevice,
                data.stream);
        if (e!=cudaSuccess)
            throw CudaException("Error while submitting async memcpy",e);
                
    }
   }
unsigned int Equals::calculateArraySize(ArrayDimensions& d, StoreType &type)
{
    string stype;
    int element_size;
    int ret;
    switch (type)
    {
        case real_float:
            element_size=sizeof(float);
            break;
        case complex_float:
            element_size=sizeof(cuFloatComplex);
            break;
        default:
            element_size=0;
    }
    ret=d.getTotalNoOfElements()*element_size;
    return ret;
}

