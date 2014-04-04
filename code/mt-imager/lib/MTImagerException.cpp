/* MTImagerException.cpp:  Implementation of the MTImagerException class 
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


#include "MTImagerException.h"
#include <cstdio>

using namespace std;
using namespace GAFW;
using namespace mtimager;
MTImagerException::MTImagerException(std::string desc,Identity * object_pointer,int linenumber, std::string file ) throw():GAFWException(desc, object_pointer,linenumber,file) 
{
    char  linestring [20];
    sprintf (linestring,"%i",linenumber);
    this->description=desc;
    this->linenumber=linestring;
    this->file=file;
    Identity* object=reinterpret_cast<Identity *>(object_pointer);
        this->object_name=object->getName();
        this->object_objectName=object->getObjectName();
}
 
MTImagerException::MTImagerException(std::string desc,void * object_pointer,int linenumber, std::string file ) throw():GAFWException(desc, object_pointer,linenumber,file)
{
    char  linestring [20];
    sprintf (linestring,"%i",linenumber);
    this->description=desc;
    this->linenumber=linestring;
    this->file=file;
    this->object_name="Not Supported";
    this->object_objectName="Not Supported";
    
}


;
MTImagerException::~MTImagerException() throw()
{
    
}

const char* MTImagerException::what() const throw()
{
    string &s=*new string("\nDescription: ");
    s+=this->description;
    s+="\nType: Imager Exception\nObject Name: ";
    s+=this->object_name;
    s+="\nObject ObjectName: ";
    s+=this->object_objectName;
    s+="\nFile Name: ";
    s+=this->file;
    s+="\nLine Number: ";
    s+=this->linenumber;
    s+="\n";
    return s.c_str();
    
}