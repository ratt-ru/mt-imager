/* BugException.cpp: Implementation of the BugException class   
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
#include "gafw.h"
using namespace GAFW;
BugException::BugException(std::string desc,Identity *object_pointer, int linenumber, std::string file ) throw(): 
        GAFWException(desc,object_pointer, linenumber,file) 
{}
BugException::BugException(std::string desc,void * object_pointer,int linenumber, std::string file ) throw():
        GAFWException(desc,object_pointer, linenumber,file) 
{}
BugException::~BugException() throw(){
}
const char* BugException::what() const throw()
{
    string &s=*new string("Type: BUG. This Exception is thrown if a BUG is detected\nDescription: ");
    s+=this->description;
    s+="\nObject Name: ";
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
