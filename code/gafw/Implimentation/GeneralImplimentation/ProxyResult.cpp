/* ProxyResult.cpp:  ProxyResult General Implementation.
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
using namespace std;
ProxyResult::ProxyResult(Factory *factory, std::string nickname,Module * mod): GAFW::ProxyResult(factory,nickname,"ProxyResultMatrix")
{
    this->module=mod;
    this->currentBind=NULL;
    this->last_snapshot_id=-1;
}
ProxyResult::~ProxyResult()
{
    //TODO
}
ProxyResult::ProxyResult()
{
    throw Bug("The constructor ProxyResult::ProxyResult() is only available for programming convenience. it should never be called");
    
}
void ProxyResult::snapshot_taken(int snapshot_no)
{
    if (this->last_snapshot_recorded!=snapshot_no)
    {
        last_snapshot_recorded=snapshot_no;
        if (this->module!=NULL)
        {
            this->module->resultRead(this,snapshot_no);
        }
       
    }
   
}
void ProxyResult::setBind(GAFW::Result *result)
{
    this->currentBind=dynamic_cast<Result *>(result);
}
GAFW::Result*  ProxyResult::getBind()
{
    return dynamic_cast<GAFW::Result *>(this->currentBind);
}
GAFW::Result * ProxyResult::retrieveResultFromId(int id)
{
    if (this->snapshot_map.count(id)!=0)
        return (this->snapshot_map[id]);
    else
        if (this->currentBind!=NULL)
            return (this->currentBind);
        else
            throw GeneralException("Snapshot id is unknown and ProxyMatrix not yet bound");
}
ArrayDimensions ProxyResult::getDimensions(CalculationId id)
{
    return this->retrieveResultFromId(id)->getDimensions(id);
}
GAFW::DataTypeBase ProxyResult::getType(CalculationId id)
{
    return this->retrieveResultFromId(id)->getType(id);
}
void ProxyResult::getValue(CalculationId id,std::vector<unsigned int> &position, ValueWrapperBase &value)
{
    this->retrieveResultFromId(id)->getValue(id,position,value);
}
void ProxyResult::getSimpleArray(CalculationId id, PointerWrapperBase &values)
{
    this->retrieveResultFromId(id)->getSimpleArray(id,values);
}
void ProxyResult::getValues(CalculationId id,PointerWrapperBase &value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,...)
{
    va_list args;
    va_start(args,noOfIndexes);
    this->retrieveResultFromId(id)->getValues(id,value,no_of_elements,isColumnWise,noOfIndexes,args);
    va_end(args);

}
CalculationId ProxyResult::calculate()
{
    if (this->currentBind==NULL) throw ValidationException("ProxyResult is not yet bound to another Result object");
    CalculationId i=this->currentBind->calculate();
    if (i>-1) {
        this->snapshot_map[i]=this->currentBind;
        this->last_snapshot_id=i;
    }
    return i;
}
CalculationId ProxyResult::getLastSnapshotId()
{
    if (this->last_snapshot_id!=-1) return this->last_snapshot_id;
    else
        if (this->currentBind==NULL) throw ValidationException("No known snapshots and ProxyResult is not bound")
        else return this->currentBind->getLastSnapshotId();
}
bool ProxyResult::isDataValid(CalculationId id)
{
    return this->retrieveResultFromId(id)->isDataValid(id);
}
void ProxyResult::requireResults()
{
    if (this->currentBind==NULL) throw GeneralException("Proxy Result not bound");
    this->currentBind->requireResults();
}
void ProxyResult::doNotRequireResults()
{
    if (this->currentBind==NULL) throw GeneralException("Proxy Result not bound");
    this->currentBind->doNotRequireResults();
}
bool ProxyResult::areResultsRequired()
{
    if (this->currentBind==NULL) throw GeneralException("Proxy Result not bound");
    this->currentBind->doNotRequireResults();
}

bool ProxyResult::waitUntilDataValid(CalculationId id)
{
      return this->retrieveResultFromId(id)->waitUntilDataValid(id);

}
void ProxyResult::removeReusabilityOnNextUse()
{
       if (this->currentBind==NULL) throw GeneralException("Proxy Result not bound");
       this->currentBind->removeReusabilityOnNextUse();
}

void ProxyResult::reusable()
{
       if (this->currentBind==NULL) throw GeneralException("Proxy Result not bound");
       this->currentBind->reusable();
    
}
void ProxyResult::notReusable()
{
       if (this->currentBind==NULL) throw GeneralException("Proxy Result not bound");
       this->currentBind->notReusable();
    
}
bool ProxyResult::isReusable()
{
       if (this->currentBind==NULL) throw GeneralException("Proxy Result not bound");
       this->currentBind->notReusable();

}
GAFW::Array *ProxyResult::getParent()
{
       if (this->currentBind==NULL) throw GeneralException("Proxy Result not bound");
       return this->currentBind->getParent();

}
void ProxyResult::overwrite()
{
       if (this->currentBind==NULL) throw GeneralException("Proxy Result not bound");
       this->currentBind->overwrite();

}
void ProxyResult::doNoOverwrite()
{
       if (this->currentBind==NULL) throw GeneralException("Proxy Result not bound");
       this->currentBind->doNoOverwrite();
   
}
bool ProxyResult::isOverwrite()
{
       if (this->currentBind==NULL) throw GeneralException("Proxy Result not bound");
       return this->currentBind->isOverwrite();
    
}
void ProxyResult::getValues(PointerWrapperBase &value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,...)
{
    if (this->currentBind==NULL) throw GeneralException("Proxy Result not bound");
    CalculationId id=currentBind->getLastSnapshotId();   
    va_list args;
    va_start(args,noOfIndexes);
    this->retrieveResultFromId(id)->getValues(id,value,no_of_elements,isColumnWise,noOfIndexes,args);
    va_end(args);
}
