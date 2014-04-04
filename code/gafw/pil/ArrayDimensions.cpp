/* ArrayDimensions.cpp
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
 * Author's contact details can be found on url http://www.danielmuscat.com 
 */
#include "gafw.h"
#include <sstream>
using namespace GAFW;
using namespace std;
ArrayDimensions::ArrayDimensions(int noOfDims, int dim0,int dim1,int dim2, int dim3,int dim4)
{
    int * dimensions[5]={ &dim0,&dim1,&dim2,&dim3,&dim4};
    
    for (int i=0;i<noOfDims;i++)
    {
        if (i<5)
            this->dims.push_back(*dimensions[i]);
        else
            this->dims.push_back(0);
    }
}
ArrayDimensions::ArrayDimensions(const ArrayDimensions& orig)
{
    this->copy(orig);
}
ArrayDimensions::ArrayDimensions()
{
}
ArrayDimensions::~ArrayDimensions()
{
    //nothing to do
}
void ArrayDimensions::setNoOfDimensions(int dim)
{
    for(;this->dims.size()!=dim;)
    {
        if (this->dims.size()<dim)
            this->dims.push_back(0);
        else
            this->dims.pop_back();
    }
            
}
int ArrayDimensions::getNoOfDimensions()
{
    return this->dims.size();
}
void ArrayDimensions::setDimension(int dimId,int value)
{
    if ((dimId<0)||(dimId>=dims.size())) throw GeneralException("Dimension index outside range");
    if (value<0) throw GeneralException("Dimension value cannot be negative");
    this->dims[dimId]=value;
}
int ArrayDimensions::getDimension(int dimId)
{
    if ((dimId<0)||(dimId>=dims.size())) throw GeneralException("Dimension index outside range");
    return this->dims[dimId];
}
bool ArrayDimensions::operator==(ArrayDimensions &other)
{
    //First check if the two are well defined. If any of them is not then function
   // will always return false
    if ((!this->isWellDefined())||(!other.isWellDefined()))
        return false;
    if (this->dims.size()!=other.dims.size())
        return false;
    for (int i=0; i<this->dims.size(); i++)
        if (this->dims[i]!=other.dims[i])
            return false;
    return true;
    
}
ArrayDimensions & ArrayDimensions::operator=(ArrayDimensions &orig )
{
    copy(orig);
    return *this;
}
bool ArrayDimensions::isWellDefined()
{
    // We define well defenition by having a greater the  0 noOfDimensions and 
    // each dim to be greater then 0
    if (this->dims.size()==0) return false;
    for (vector<int>::iterator i=this->dims.begin(); i<this->dims.end(); i++)
    {
        if (*i<1) return false; 
    }
    return true;
}

std::string ArrayDimensions::logString()
{
    stringstream toReturn;
    
    toReturn<<"No Of Dimensions=";
    toReturn<<this->dims.size();
    for (int x=0; x<this->dims.size(); x++)
    {
        toReturn<<" Dim"<<x << "=" <<this->dims[x];
    }
    return toReturn.str();
    
}
void ArrayDimensions::copy(const ArrayDimensions& orig)
{
    //Check if there is a lock and if so ensure NoOfDims is not changed
    this->dims=orig.dims;
}
unsigned int ArrayDimensions::getTotalNoOfElements()
{
    int ret=1;
    for (vector<int>::iterator i=this->dims.begin();i<this->dims.end();i++)
    {
        ret*=*i;
    }
    return ret;
}
bool ArrayDimensions::isPositionInArray(std::vector<unsigned int> & position)
{
    if (!this->isWellDefined()) return false;
    if (position.size()!=this->dims.size()) return false;
    for (int i=0;i<position.size();i++)
    {
        if (position[i]>=this->dims[i]) return false;
        
    }
    
}
ArrayDimensions::operator std::string()
{
    stringstream out;
    bool first=true;
    for (vector<int>::iterator i=dims.begin();i<dims.end();i++)
    {
        if (!first) out <<'x';
        out<< *i;
        first=false;
    }
    return string(out.str());
}

       
