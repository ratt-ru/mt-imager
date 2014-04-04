/* OperationDescriptor.cpp:  Implementation of the  OperationDescriptor class. 
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

using namespace GAFW::GPU;
OperationDescriptor::OperationDescriptor()
{
    this->buffer.GPUPointer=NULL;
    
    this->buffer.mode=GAFW::GPU::Buffer::Buffer_Normal;
    this->buffer.size=0;
    this->specialActions=NoAction;
    this->noOfInputs=0;
    this->inputs=NULL;
    this->outputs=NULL;
    this->noOfOutputs=0;
    this->eventEnd=NULL;
    this->eventStart=NULL;
   
}
OperationDescriptor::~OperationDescriptor() 
{
    if (this->noOfInputs!=0)
        delete [] this->inputs;
    if (this->noOfOutputs!=0)
        delete [] this->outputs;
}

