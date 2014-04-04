/* Result.h:  Definition of the general implementation of the Result
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
#ifndef __GEN_RESULT_H__
#define	__GEN_RESULT_H__
#include "gafw-impl.h"
#include <map>
namespace GAFW { namespace GeneralImplimentation {

class Result:public GAFW::Result
{
private:
    Result(const Result& m){};
protected:
    Result();
    Array *output_of;
    std::map<int,DataStore*> stores;
    //std::map<int,ResultStatus> status;
    bool resultsRequired;
    bool toReuse;
    bool toOverwrite;
    bool removeReusability;
    Result(Factory *factory, Array * parent);
    ~Result();
    DataStore * getStore(int id);
//    void setStatus(int id, ResultStatus);
    void createStore(int id,DataTypeBase &type, ArrayDimensions &dim);
    
    friend class Engine; //will access the above 3 functions
    friend class Factory;
public:
    //void getValues(CalculationId id,PointerWrapperBase &value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,va_list &vars);
    virtual ArrayDimensions getDimensions(CalculationId id);
    virtual void getValue(CalculationId id,std::vector<unsigned int> &position, ValueWrapperBase &value);
    virtual void getSimpleArray(CalculationId id, PointerWrapperBase &values);
    virtual void getValues(CalculationId id,PointerWrapperBase  &value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,...);
    virtual void getValues(PointerWrapperBase &value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,...);
    virtual DataTypeBase getType(CalculationId id=-1);
    virtual void requireResults();
    virtual void doNotRequireResults();
    virtual bool areResultsRequired();
    virtual bool isDataValid(CalculationId id=-1);
    virtual bool waitUntilDataValid(CalculationId id=-1);
    
    
    virtual CalculationId calculate();
    virtual void removeReusabilityOnNextUse();
    virtual void reusable();
    virtual void notReusable();
    virtual bool isReusable();
    virtual Array *getParent();
    virtual void overwrite();
    virtual void doNoOverwrite();
    virtual bool isOverwrite();
    virtual CalculationId getLastSnapshotId();
   
};




} }

#endif	/* RESULTMATRIX_H */

