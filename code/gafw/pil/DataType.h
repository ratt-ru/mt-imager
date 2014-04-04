/* DataType.h:  Definition of DataType and DataTypeBase classes     
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
#ifndef __PIL_DATATYPE_H__
#define	__PIL_DATATYPE_H__
namespace GAFW
{
    class DataTypeBase
    {
    public:
        int _type; 
       
    protected:
        inline DataTypeBase(const int _type ):_type(_type){};
    };

    template <class T>
    class DataType: public DataTypeBase
    {

    public:
        DataType();
    };
    
    
    
}

#endif	

