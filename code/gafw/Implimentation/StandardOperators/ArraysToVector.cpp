/* ArraysToVector.cpp:  Pure C++ implementation of the ArraysToVector operator 
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
#include "cuComplex.h"
#include "ArraysToVector.h"
#include <sstream>

using namespace GAFW::GeneralImplimentation;
using namespace GAFW::GPU::StandardOperators;

ArraysToVector::ArraysToVector(GAFW::GPU::GPUFactory * factory,std::string nickname) : GPUArrayOperator(factory,nickname,string("Operator::ArraysToVector"))
{
    
}
ArraysToVector::~ArraysToVector()
{

}

void ArraysToVector::validate()
{
    
   //No of inputs
    if (this->inputs.size()==0) 
    {
        throw ValidationException("At least 1 input is expected");
    }
    
    if (this->outputs.size()!=1) throw ValidationException("Only one output is supported")
    
    int noOfInputs=inputs.size();
    
    if (!this->inputs[0]->isDefined()) throw ValidationException("One of the inputs is not defined");
    StoreType type=this->inputs[0]->getType();
    unsigned int totalElements=0; 
    //check that all inputs are defined and add the total no of elements
    for (int i=0;i<noOfInputs;i++)
    {
        if (!this->inputs[i]->isDefined()) throw ValidationException("Input is not defined");
        if (this->inputs[i]->getType()!=type) throw ValidationException("All inputs must be of the same StoreType");
        totalElements+=this->inputs[i]->getDimensions().getTotalNoOfElements();
    }
    //Seems all input is as it should be and we know the total amount of elements
    ArrayDimensions arr(1,totalElements);
    this->outputs[0]->setDimensions(arr);
    this->outputs[0]->setType(type);
}

void ArraysToVector::submitToGPU(GAFW::GPU::GPUSubmissionData &data)
{
    //We do not need to launch an specific kernel as e can use memcpy
    
    int elementSize;
    switch (data.outputs->type)
    {
        case complex_float:
            elementSize=sizeof(cuComplex);
            break;
        case real_float:
            elementSize=sizeof(float);
            break;
        case real_double:
            elementSize=sizeof(double);
            break;
        default:
            throw GeneralException("Unknown type");
    }
    void * begin_write_pointer=data.outputs->pointer;
    for (int i=0;i<data.noOfInputs;i++)
    {
        int noOfElements=data.inputs[i].dim.getTotalNoOfElements();
        cudaError_t cudaError;  
        cudaError=cudaMemcpyAsync(begin_write_pointer,data.inputs[i].pointer,noOfElements*elementSize,cudaMemcpyDeviceToDevice,data.stream);
        if (cudaError!=cudaSuccess) throw CudaException("Unable to copy on device",cudaError);
        begin_write_pointer+=noOfElements*elementSize;
    }
    
        //That's all
}
