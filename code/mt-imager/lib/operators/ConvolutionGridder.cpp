/* ConvolutionGridder.cpp:  Pure C++ implementation of the ConvolutionGridder operator 
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
using namespace GAFW::GPU;

ConvolutionGridder::ConvolutionGridder(GPUFactory * factory,std::string nickname) : GPUArrayOperator(factory,nickname,string("Operator::ConvGridder"))
{
    
}
ConvolutionGridder::~ConvolutionGridder()
{

}

void ConvolutionGridder::validate(){
    /********************************************************************
     * VAlidations:                                                     *
     * 1. 1 input Matrix which must have valid dimensions
     * 1. Only one inputs and one output is supported
     * 
     * Output matrix will be set to same dimensions and store type
     * of input matrix
     * ******************************************************************/
  
   //No of inputs
    if (this->inputs.size()<6) 
    {
        throw ValidationException("At least Six inputs  (others fake)");
    }
    
    if (this->outputs.size()!=1) throw ValidationException("Only one output is supported");
    
    
    /* To do... if all is right then we do not defne outpir*/
}
void ConvolutionGridder::postRunExecute(void* texObj)
{
    cudaDestroyTextureObject(*((cudaTextureObject_t*)(texObj))); //,"Unable to destroy texture");
}


