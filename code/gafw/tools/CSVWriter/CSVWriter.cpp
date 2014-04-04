/* CSVWriter.cpp: Code for the CSVWriter class, part of the GAFW CSVWriter Tool     
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

#include <vector>
#include "CSVWriter.h"
using namespace GAFW::Tools::CSVWriter;
CSVWriter::CSVWriter(std::ostream * stream,const char * delimeter)
{
    this->myStream=stream;
    this->delimeter=delimeter;
    this->streamControlledByMyself=false;
}
CSVWriter::CSVWriter(const char * filename,const char * delimeter)
{
    std::ofstream *st=new std::ofstream();
    this->myStream=st;
    st->open(filename);
    
    if (!st->is_open())
        throw CSVWriterException("CSVWriter::CSVWriter()",std::string("Unable to open file for write: ") + filename);
    this->delimeter=delimeter;
    this->streamControlledByMyself=true;
}


CSVWriter::~CSVWriter()
{
    if (this->streamControlledByMyself) {
        this->myStream->flush();
        delete this->myStream;
    }
}
void CSVWriter::writeTitle()
{
    std::vector<CSVColumnBase *>::iterator i=this->cols.begin();
    (*myStream)<<(*i)->getTitle();
    i++;
    for (;i<cols.end();i++)
        (*myStream)<<this->delimeter<<(*i)->getTitle();
    (*myStream) <<std::endl;
    myStream->flush();
}
void CSVWriter::writeRow()
{
     std::vector<CSVColumnBase *>::iterator i=this->cols.begin();
    (*myStream)<<(*i)->getValue();
    i++;
    for (;i<cols.end();i++)
        (*myStream)<<this->delimeter<<(*i)->getValue();
    (*myStream) <<std::endl;
    myStream->flush();
}
void CSVWriter::addColumn(CSVColumnBase * col)
{
    this->cols.push_back(col);
}
      
