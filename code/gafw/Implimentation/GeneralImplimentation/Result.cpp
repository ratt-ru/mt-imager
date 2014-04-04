/* Result.cpp: General Implementation of the Result
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
using namespace GAFW::GeneralImplimentation;
Result::Result(Factory *factory, Array * parent):GAFW::Result(factory,string("Result of ")+parent->objectName,"Result")
{
    this->output_of=parent;
    this->resultsRequired=false;
    this->toReuse=false;
    this->toOverwrite=false;
    this->stores.clear();
    this->removeReusability=false;
}
Result::Result()
{
    throw Bug("The constructor Result::Result() is available for programming convenience only and should never be called");
}
Result::~Result()
{
    
    //TODO
}
void Result::removeReusabilityOnNextUse()
{
    this->removeReusability=true;
}
DataStore * Result::getStore(int id)
{
    if (this->stores.count(id)==0) return NULL;
    return this->stores[id];
    
}

void Result::createStore(int id, DataTypeBase &type, ArrayDimensions &dim)
{
    //We first check if store is already created
    if (this->stores.count(id)!=0) throw GeneralException("Store for the requested id has already been created");
    DataStore *m=dynamic_cast<Factory*>(this->getFactory())->requestDataStore(type, dim,false);
    if (m==NULL) throw GeneralException("Store was not created");
    this->stores[id]=m;
}
ArrayDimensions Result::getDimensions(CalculationId id)
{
    
    if (this->stores.count(id)==0)  throw GeneralException("No store with the requested id was found");
   return this->stores[id]->getDimensions();
    
}
void Result::requireResults()
{
    this->resultsRequired=true;
}
void Result::doNotRequireResults()
{
    this->resultsRequired=false;
}
CalculationId Result::calculate()
{
    return dynamic_cast<Factory *>(this->getFactory())->requestEngine()->calculate(this);
}
bool Result::areResultsRequired()
{
    return this->resultsRequired;
}
void Result::reusable()
{
    this->toReuse=true;
}
void Result::notReusable()
{
    this->toReuse=false;
}
bool Result::isReusable()
{
    return this->toReuse;
}
bool Result::isOverwrite()
{
    return this->toOverwrite;
}
Array * Result::getParent()
{
    return this->output_of;
}
DataTypeBase Result::getType(CalculationId id)
{
    if (this->stores.count(id)==0) return DataType<void>();
    return this->stores[id]->getType();
    
}

void Result::overwrite()
{
    this->toOverwrite=true;
}
void Result::doNoOverwrite()
{
    this->toOverwrite=false;
}
CalculationId Result::getLastSnapshotId()
{
    
    if (this->stores.size()!=0)
    {
        int i= this->stores.rbegin()->first;
        return i;
    }
    else 
        throw GeneralException("No snapshot is currently known");
}

bool Result::isDataValid(CalculationId id)
{
    if (id==-1) id=this->getLastSnapshotId();
    return this->stores[id]->isDataValid();
}
bool Result::waitUntilDataValid(CalculationId id)
{
    if (id==-1) id=this->getLastSnapshotId();
     this->stores[id]->waitUntilDataValid();
}
void Result::getValues(PointerWrapperBase & value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,...)
{
    va_list args;
    va_start(args,noOfIndexes);
    this->stores[this->getLastSnapshotId()]->getValues(value,no_of_elements,isColumnWise,noOfIndexes,args);
    va_end(args);
}
    
void Result::getValue(CalculationId id,std::vector<unsigned int> &position, ValueWrapperBase &value)
{
    if (this->stores.count(id)==0) throw GeneralException("Calculation ID requested does not exist");
    this->stores[id]->getValue(position,value);
}
void Result::getSimpleArray(CalculationId id, PointerWrapperBase &values)
{
        if (this->stores.count(id)==0) throw GeneralException("Calculation ID requested does not exist");
        this->stores[id]->getSimpleArray(values);
}
void Result::getValues(CalculationId id,PointerWrapperBase  &value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,...)
{
    if (this->stores.count(id)==0) throw GeneralException("Calculation ID requested does not exist");
    va_list args;
    va_start(args,noOfIndexes);
    this->stores[id]->getValues(value,no_of_elements,isColumnWise,noOfIndexes,args);
    va_end(args);
}
    