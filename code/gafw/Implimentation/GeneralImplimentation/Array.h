/* Array.h:  Header file for the General Implementation of the GAFW Array.
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

#ifndef __GEN_ARRAY_H__
#define	__GEN_ARRAY_H__
#include <string>
#include <complex>
#include <stdarg.h>

namespace GAFW { namespace GeneralImplimentation 
{
    
    
class Array :public GAFW::Array {
private:
    Array(const Array& orig):type(DataType<void>()){}; //copies are disallowed.. 
    DataTypeBase storeTypeToTypeConvert(int _type);
protected:
    std::vector <ArrayOperator*> input_to;  //Can be an input to many operators  
    std::vector <Array*> preValidatorDependents;
    ArrayOperator *output_of;   ///but we only allow to be an output of only one operator
    std::vector<DataStore *> oldStores;
    DataStore * store; //This is the current store
    Result *result;
    GAFW::Result *result_Outputof;
    PreValidator *preValidator;
    
    DataTypeBase type;
    Array(Factory *factory,std::string nickname); //creation and destruction is a function of the factory
    ~Array();
    ArrayDimensions * tmp_dim;
    friend class ArrayOperator;
    friend class Factory;
    friend class Engine;
    //A helping function
    inline void createStoreIfNotExist(ValueWrapperBase& input);
    inline void createStoreIfNotExist(PointerWrapperBase &input);

public:
    virtual void createStore();
    virtual void createStore(DataTypeBase &dataType);
    
    //virtual void setDimensions(ArrayDimensions &dim);
    virtual void setValue(ValueWrapperBase &value, unsigned int NoOfIndexes,...);
    virtual void setValues(PointerWrapperBase &pointer, int no_of_elements,bool isColumnWise, unsigned int NoOfIndexes,...);
    virtual void setValue(std::vector<unsigned int> &position, ValueWrapperBase &value);
    virtual void setAllValues(std::vector<unsigned int> &position, PointerWrapperBase &valuesPointer);
    
    virtual ArrayDimensions getDimensions();
    virtual GAFW::Result * getResults();
    virtual bool isDefined();
    virtual void bind_to(GAFW::Result *m);
    virtual void preValidateWith(PreValidator *preValidator);
    virtual void clearPreValidatorDependents();
    virtual void setPreValidatorDependent(GAFW::Array *m);
    virtual void setPreValidatorDependent(std::string arrayObjectName);
    StoreType getType();
    void setType(StoreType type);
    void createStore(StoreType);
    virtual void createStore(DataTypeBase dataType);
    virtual void setDimensions(ArrayDimensions dim);
    virtual void setType(DataTypeBase dataType);
};

//inline functions
void inline Array::setValue(std::vector<unsigned int> &position, ValueWrapperBase &value)
{ 
        this->createStoreIfNotExist(value);    
        this->store->setValue(position,value);
}
void inline Array::setValue(ValueWrapperBase &value, unsigned int NoOfIndexes,...)
{
    this->createStoreIfNotExist(value);
    va_list args;
    va_start(args,NoOfIndexes);
    this->store->setValue(value,NoOfIndexes,args);
    va_end(args);
}
    
void inline Array::setValues(PointerWrapperBase &values, int no_of_elements,bool isColumnWise, unsigned int NoOfIndexes,...)
{
    this->createStoreIfNotExist(values);
    va_list args;
    va_start(args,NoOfIndexes);
    this->store->setValues(values,no_of_elements,isColumnWise,NoOfIndexes,args);
    va_end(args);
}
inline void Array::createStoreIfNotExist(ValueWrapperBase &input)
{
   if (store==NULL)
   {    if (this->type._type==StoreTypeUnknown)
        this->type=this->storeTypeToTypeConvert(ValuePointerTypeMap[input.valueType]);
        this->createStore();
   }
}
inline void Array::createStoreIfNotExist(PointerWrapperBase &input)
{
   if (store==NULL)
   {    if (this->type._type==StoreTypeUnknown)
        this->type=this->storeTypeToTypeConvert(ValuePointerTypeMap[input.pointerType]);
        this->createStore();
   }
}



}}//End of namespace


#endif	

