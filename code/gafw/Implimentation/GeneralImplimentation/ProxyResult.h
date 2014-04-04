/* ProxyResult.h:  Definition of ProxyResult General Implementation.
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
#ifndef __GEN_PROXYRESULT_H__
#define	__GEN_PROXYRESULT_H__
#include "gafw-impl.h"
#include <map>
namespace GAFW { namespace GeneralImplimentation {

    

class ProxyResult:  public GAFW::ProxyResult  {
private:
    ProxyResult(const ProxyResult& orig){};
protected:
    ProxyResult();
    ProxyResult(Factory * factory, std::string nickname,Module * mod);
    virtual ~ProxyResult();
   
    Result * currentBind;
    Module * module;     //The module that this is a member of
    std::map<int,Result*> snapshot_map;
    Result * retrieveResultFromId(int id);
    int last_snapshot_recorded;
    int last_snapshot_id;
    friend class Factory;
public:
    void snapshot_taken(int snapshot_no);
    void setBind(GAFW::Result *);
    GAFW::Result*  getBind();
    
    virtual ArrayDimensions getDimensions(CalculationId id);
    virtual void getValue(CalculationId id,std::vector<unsigned int> &position, ValueWrapperBase &value);
    virtual void getSimpleArray(CalculationId id, PointerWrapperBase &values);
    virtual void getValues(CalculationId id,PointerWrapperBase &value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,...);
    virtual DataTypeBase getType(CalculationId id=-1);
    virtual void requireResults();
    virtual void doNotRequireResults();
    virtual bool areResultsRequired();
    virtual bool isDataValid(CalculationId id=-1);
    virtual bool waitUntilDataValid(CalculationId id=-1);
    virtual void getValues(PointerWrapperBase &value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,...);
    
    virtual CalculationId calculate();
    virtual void removeReusabilityOnNextUse();
    virtual void reusable();
    virtual void notReusable();
    virtual bool isReusable();
    virtual GAFW::Array *getParent();
    virtual void overwrite();
    virtual void doNoOverwrite();
    virtual bool isOverwrite();
    virtual CalculationId getLastSnapshotId();


    
};
}} //end of namespace
#endif	/* COPYRESULTMATRIX_H */

