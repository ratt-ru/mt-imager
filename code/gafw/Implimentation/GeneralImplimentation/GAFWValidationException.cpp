/* GAFWValidationException.cpp:  Implementation of the GAFWValidationException and macros ValidationException,ValidationException2 
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

#include "gafw-impl.h"
#include <cstdio>

using namespace std;
using namespace GAFW::GeneralImplimentation;
 
GAFWValidationException::GAFWValidationException(std::string desc,Identity * object_pointer,int linenumber, std::string file ):GAFWException(desc, object_pointer,linenumber,file)
{
    /*char  linestring [20];
    sprintf (linestring,"%i",linenumber);
    this->description=desc;
    this->linenumber=linestring;
    this->file=file;
    //If so object_name and object_nickname should be changed
        Identity* object=reinterpret_cast<Identity *>(object_pointer);
        this->object_name=object->getName();
        this->object_nickname=object->getNickname();
   */     
}
GAFWValidationException::GAFWValidationException(std::string desc,void * object_pointer,int linenumber, std::string file ):GAFWException(desc, object_pointer,linenumber,file)
{
    /*
    //object_pointer is ignored in this case
    char  linestring [20];
    sprintf (linestring,"%i",linenumber);
    this->description=desc;
    this->linenumber=linestring;
    this->file=file;
    //If so object_name and object_nickname should be changed
    this->object_name="Not Supported";
    this->object_nickname="Not Supported";
    */    
}

GAFWValidationException::~GAFWValidationException() throw()
{
    
}

const char* GAFWValidationException::what() const throw()
{
    string &s=*new string("\nDescription: ");
    s+=this->description;
    s+="\nType: Validation Exception\nObject Name: ";
    s+=this->object_name;
    s+="\nObject Objectname: ";
    s+=this->object_objectname;
    s+="\nFile Name: ";
    s+=this->file;
    s+="\nLine Number: ";
    s+=this->linenumber;
    s+="\n";
    return s.c_str();
}