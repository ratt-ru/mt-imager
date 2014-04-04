/* Array.h:  Header file for the GAFW PIL definition of an Array.
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

#ifndef __PIL_ARRAY_H__
#define	__PIL_ARRAY_H__
#include <string>
#include <complex>
#include <stdarg.h>
namespace GAFW 
{
class Array :public FactoryAccess, public Identity, public LogFacility {
private:
    Array(const Array& orig){}; //copies are disallowed.. 
protected:
    Array(); 
    inline Array(Factory *f,std::string objectName,std::string name);
public:
    virtual void createStore()=0;
    virtual void createStore(DataTypeBase dataType)=0;
    virtual void setDimensions(ArrayDimensions dim)=0;
    virtual void setValue(ValueWrapperBase &value, unsigned int NoOfIndexes,...)=0;
    virtual void setValues(PointerWrapperBase &pointer, int no_of_elements,bool isColumnWise, unsigned int NoOfIndexes,...)=0;
    virtual void setValue(std::vector<unsigned int> &position, ValueWrapperBase &value)=0;
    virtual void inline setAllValues(std::vector<unsigned int> &position, PointerWrapperBase &valuesPointer)=0;
    virtual void setType(DataTypeBase dataType)=0;
    virtual ArrayDimensions getDimensions()=0;
    virtual Result * getResults()=0;
    virtual bool isDefined()=0;
    virtual void bind_to(Result *m)=0;
    virtual void preValidateWith(PreValidator *preValidator)=0;
    virtual void clearPreValidatorDependents()=0;
    virtual void setPreValidatorDependent(Array *m)=0;
    virtual void setPreValidatorDependent(std::string arrayObjectName)=0;
};

inline Array::Array(Factory *f, std::string objectName,std::string name):FactoryAccess(f),Identity(objectName,name)
{
    
}

}//End of namespace


#endif	

