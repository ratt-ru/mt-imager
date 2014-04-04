/* cuFFTException.cpp:  Implementation of the cuFFTException class. 
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

#include "cuFFTException.h"
#include <cufft.h>
#include <string>
//#include <sstream>
using namespace std;
using namespace GAFW;
using namespace GAFW::GPU::StandardOperators;        
 
cuFFTException::cuFFTException(std::string desc,cufftResult error,Identity * object,int linenumber, std::string file ):GAFWException(desc, object,linenumber,file) 
{
    char  linestring [20];
    sprintf (linestring,"%i",linenumber);
    this->description=desc;
    this->linenumber=linestring;
    this->file=file;
    this->error=error;
    //First lets clear out identity
    this->object_name=object->getName();
    this->object_objectname=object->getObjectName();
    
    //Now we need to decode the error
    switch (error) 
    {
        case CUFFT_SUCCESS:
            this->errorName="CUFFT_SUCCESS";
            break;
        case CUFFT_INVALID_PLAN:
            this->errorName="CUFFT_INVALID_PLAN";
            break;
        case CUFFT_ALLOC_FAILED:
            this->errorName="CUFFT_ALLOC_FAILED:";
            break;
        case CUFFT_INVALID_TYPE:
            this->errorName="CUFFT_INVALID_TYPE";
            break;
        case CUFFT_INVALID_VALUE:
            this->errorName="CUFFT_INVALID_VALUE";
            break;
            
        case CUFFT_INTERNAL_ERROR:
            this->errorName="CUFFT_INTERNAL_ERROR";
            break;
        case CUFFT_EXEC_FAILED:
            this->errorName="CUFFT_EXEC_FAILED";
            break;
        case CUFFT_SETUP_FAILED:
            this->errorName="CUFFT_SETUP_FAILED";
            break;
        case CUFFT_INVALID_SIZE:
            this->errorName="CUFFT_INVALID_SIZE";
            break;
        default:
            this->errorName="Unknown";
            
                    
    
    }
    
    
    
}
cuFFTException::cuFFTException(std::string desc,cufftResult error,void * object,int linenumber, std::string file ):GAFWException(desc, object,linenumber,file)
{
    char  linestring [20];
    sprintf (linestring,"%i",linenumber);
    this->description=desc;
    this->linenumber=linestring;
    this->file=file;
    this->error=error;
    this->object_name="Not supported";
    this->object_objectname="Not supported";
    
    //Now we need to decode the error
    switch (error) 
    {
        case CUFFT_SUCCESS:
            this->errorName="CUFFT_SUCCESS";
            break;
        case CUFFT_INVALID_PLAN:
            this->errorName="CUFFT_INVALID_PLAN";
            break;
        case CUFFT_ALLOC_FAILED:
            this->errorName="CUFFT_ALLOC_FAILED:";
            break;
        case CUFFT_INVALID_TYPE:
            this->errorName="CUFFT_INVALID_TYPE";
            break;
        case CUFFT_INVALID_VALUE:
            this->errorName="CUFFT_INVALID_VALUE";
            break;
            
        case CUFFT_INTERNAL_ERROR:
            this->errorName="CUFFT_INTERNAL_ERROR";
            break;
        case CUFFT_EXEC_FAILED:
            this->errorName="CUFFT_EXEC_FAILED";
            break;
        case CUFFT_SETUP_FAILED:
            this->errorName="CUFFT_SETUP_FAILED";
            break;
        case CUFFT_INVALID_SIZE:
            this->errorName="CUFFT_INVALID_SIZE";
            break;
        default:
            this->errorName="Unknown";
    }
            
}
cuFFTException::~cuFFTException() throw()
{
    
}

const char* cuFFTException::what() const throw()
{
    string &s= *new string("\nDescription: ");
    s+=this->description;
    s+="\nType: cuFFT Exception\ncuFFT Error: ";
    s+=this->errorName;
    s+="\nObject Name: ";
    s+=this->object_name;
    s+="\nObject ObjectName: ";
    s+=this->object_objectname;
    s+="\nFile Name: ";
    s+=this->file;
    s+="\nLine Number: ";
    s+=this->linenumber;
    s+="\n";
    return s.c_str();
}