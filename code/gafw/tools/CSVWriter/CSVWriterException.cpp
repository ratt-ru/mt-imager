/* CSVWriterException.cpp: Code for the CSVWriterException class, part of the GAFW CPPProperties Tool     
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
 * Author's contact details can be found on url http://www.danielmuscat.com 
 */

#include "CSVWriterException.h"

using namespace std;
using namespace GAFW::Tools::CSVWriter;

CSVWriterException::CSVWriterException(std::string function,std::string reason) throw()
{
    this->function=function;
    
    this->reason=reason;
    this->whatReturn=new string();
}
CSVWriterException::CSVWriterException(const CSVWriterException& orig) throw()
{
    this->function=orig.function;
    this->reason=orig.reason;
    this->whatReturn=new string(*orig.whatReturn);

}
CSVWriterException::~CSVWriterException() throw()
{
    
}
const char* CSVWriterException::what() const throw()
{
    this->whatReturn->clear();  //empty the string
    whatReturn->append("\nFunction called: ");
    whatReturn->append(this->function).append("\nReason: ");
    whatReturn->append(this->reason).append("\n");
    return whatReturn->c_str();
}
   