/* GPUArrayOperator.cpp:  Implementation of the GPUCudaException class which is thrown when a CUDA error occurs.
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
#include <cstdio>

using namespace std;
using namespace GAFW::GPU;
 
GPUCudaException::GPUCudaException(std::string desc,cudaError_t error,int linenumber, std::string file ) throw()
{
    char  linestring [20];
    sprintf (linestring,"%i",linenumber);
    this->description=desc;
    this->linenumber=linestring;
    this->file=file;
    this->error=error;
}

GPUCudaException::~GPUCudaException() throw()
{
    
}

const char* GPUCudaException::what() const throw()
{
    string &s=*new string("\nDescription: ");
    s+=this->description;
    s+="\nType: Cuda Exception\nCuda Error: ";
    s+=cudaGetErrorString(this->error);
    s+="\nFile Name: ";
    s+=this->file;
    s+="\nLine Number: ";
    s+=this->linenumber;
    s+="\n";
    return s.c_str();
}