/* Array.cpp:  General implementation of the GAFW Array
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
#include "DataStore.h"
#include "Array.h"
#include <string>
using namespace GAFW::GeneralImplimentation;
 Array::Array(Factory *factory,std::string nickname):GAFW::Array(factory,nickname,"Array"),type(DataType<void>())
 {
    if (nickname==string("")) this->logWarn(builder,"No nickname given to matrix. SUch functionality will be obsoleted");
    this->input_to.clear(); //Not really needed but just in case
    this->output_of=NULL;
    this->store=NULL;
    this->result=NULL;
    this->result_Outputof=NULL;
    this->preValidator=NULL;
    this->tmp_dim=new ArrayDimensions();
    this->result=dynamic_cast<Factory *>(this->getFactory())->requestResult(this);
}
 
Array::~Array() {
    //In here we have to include distruction of datastore
}
void Array::createStore(DataTypeBase &type)
{   
    //Make sure there is no store
    if (this->store!=NULL) throw GeneralException("Store already created");
    // Lets check that we can create one
    // If The matrix is already defined as getting input from a function 
    // or the matrix has undefined dimensions we can't create
    if (this->output_of!=NULL) throw GeneralException("Store cannot be created as Array is set as output");
    if (!this->tmp_dim->isWellDefined()) throw GeneralException("Dimensions not yet defined");
            
        
    //Ok let's try and create the store
    //We need to ask factory for a datastore that stores the required type
    
    this->store=dynamic_cast<Factory *>(this->getFactory())->requestDataStore(type,*tmp_dim);
    if (this->store==NULL)
        throw GeneralException("Factory did not create store");
    this->type=type;
}
void Array::createStore()
{
    //Make sure there is no store
    if (this->store!=NULL) throw GeneralException("Store already created");
    // Lets check that we can create one
    // If The matrix is already defined as getting input from a function 
    // or the matrix has undefined dimensions we can't create
    if (this->output_of!=NULL) throw GeneralException("Store cannot be created as Array is set as output");
    if (!this->tmp_dim->isWellDefined()) throw GeneralException("Dimensions not yet defined");
            
        
    //Ok let's try and create the store
    //We need to ask factory for a datastore that stores the required type
    
    this->store=dynamic_cast<GAFW::GeneralImplimentation::Factory *> (this->getFactory())->requestDataStore(type,*tmp_dim);
    if (this->store==NULL)
        throw GeneralException("Factory did not create store");
    //this->type=type;
    
}

void Array::setDimensions(ArrayDimensions dim)
{
  //if (this->store!=NULL) throw GeneralException("setDimensions():Dimensions cannot be changed since store already created");
  //*(this->tmp_dim)=dim;
  if (this->store!=NULL) {
      this->store->setUnlinkedFromArray();
      this->store=NULL; //we have to change store is this case
  }
  *(this->tmp_dim)=dim;
  
  //TODO: I need to design a way of how to see if the datastore can be removed
  
  
}

GAFW::Result * Array::getResults()
{
    if (this->result==NULL)
        this->result=dynamic_cast<Factory *>(this->getFactory())->requestResult(this);
    return this->result;
}
void Array::bind_to(GAFW::Result* m)
{//For a bind to be accepted matrix should not be the output of an operator
    // for the time being
    
    this->result_Outputof=m;
    
}
void Array::setType(DataTypeBase type)
{
    
    //To change
  if (this->store!=NULL) {
      this->store->setUnlinkedFromArray();
      this->store=NULL; //we have to change store is this case
  }
  this->type=type;
    
}

ArrayDimensions Array::getDimensions()
{
    return *this->tmp_dim;
}

bool Array::isDefined()
{
    if ((!this->tmp_dim->isWellDefined())||(this->type._type==StoreTypeUnknown)) return false;
    return true;
}
void Array::preValidateWith(PreValidator* preValidator)
{
    this->preValidator=preValidator;
}
void Array::clearPreValidatorDependents()
{
    this->preValidatorDependents.clear();
}
void Array::setPreValidatorDependent(GAFW::Array *m)
{
    this->preValidatorDependents.push_back(dynamic_cast<Array *>(m));
}
void Array::setPreValidatorDependent(std::string arrayNickname)
{
    this->preValidatorDependents.push_back(dynamic_cast<Array*>(this->getFactory()->getArray(arrayNickname)));
}
/*void Array::setDimensions(ArrayDimensions &dim)
{
     throw GeneralException("Function not implemented");

}*/

void Array::setAllValues(std::vector<unsigned int> &position, GAFW::PointerWrapperBase &valuesPointer)
{
     throw GeneralException("Function not implemented");
    
}
 StoreType Array::getType()
 {
     return (StoreType)this->type._type;
 }
 
 
 
void Array::createStore(GAFW::DataTypeBase type)
{
    this->setType(type);
    this->createStore();
    
}
void Array::setType(GAFW::GeneralImplimentation::StoreType type)
{
    this->setType(DataTypeManual(type));
    
}
DataTypeBase Array::storeTypeToTypeConvert(int type)
{
     throw GeneralException("Function not implemented");
    
}
