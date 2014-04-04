/* ArrayDimensions.h:  Header file for the GAFW PIL definition of an Array.
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

#ifndef __PIL_ARRAYDIMENSIONS_H__
#define	__PIL_ARRAYDIMENSIONS_H__

namespace GAFW
{

    class ArrayDimensions 
    {
    private:
        std::vector<int> dims;
        void copy(const ArrayDimensions& orig);
    public:
        ArrayDimensions(int noOfDims, int dim0=0,int dim1=0,int dim2=0, int dim3=0,int dim4=0);
        ArrayDimensions(const ArrayDimensions& orig);
        ArrayDimensions();
        ~ArrayDimensions();
        void setNoOfDimensions(int dim);
        int getNoOfDimensions();
        void setDimension(int dimId,int value);
        int getDimension(int dimId);
        bool operator==(ArrayDimensions &other);
        operator std::string();
        ArrayDimensions & operator=(ArrayDimensions &orig );
        bool isWellDefined();
        std::string logString();
        inline int getNoOfRows();
        inline int getNoOfColumns();
        inline int getX();
        inline int getY();
        inline int getZ();
        inline void setNoOfRows(int value);
        inline void setNoOfColumns(int value);
        inline void setX(int value);
        inline void setY(int value);
        inline void setZ(int value);
        
        
        unsigned int getTotalNoOfElements();
        bool isPositionInArray(std::vector<unsigned int> & position);
        
        
    };
    
    inline int ArrayDimensions::getNoOfRows()
    {
        return this->getDimension(this->dims.size()-2);
    }
    
    inline int ArrayDimensions::getNoOfColumns()
    {
        return this->getDimension(this->dims.size()-1);
    }

    inline int ArrayDimensions::getX()
    {
        return this->getDimension(this->dims.size()-1);
    }
    inline int ArrayDimensions::getY()
    {
        return this->getDimension(this->dims.size()-2);
    }
    inline int ArrayDimensions::getZ()
    {
        return this->getDimension(this->dims.size()-3);
    }
    inline void ArrayDimensions::setNoOfRows(int value)
    {
        return this->setDimension(this->dims.size()-2,value);
    }
    
    inline void ArrayDimensions::setNoOfColumns(int value)
    {
        return this->setDimension(this->dims.size()-1,value);
    }

    inline void ArrayDimensions::setX(int value)
    {
        return this->setDimension(this->dims.size()-1,value);
    }
    inline void ArrayDimensions::setY(int value)
    {
        return this->setDimension(this->dims.size()-2,value);
    }
    inline void ArrayDimensions::setZ(int value)
    {
        return this->setDimension(this->dims.size()-3,value);
    }
}

#endif	/* ARRAYDIMENSIONS_H */

