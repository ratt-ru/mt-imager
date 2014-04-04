/* Result.h:  Definition and skeletal code for the Module 
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

#ifndef __PIL_RESULT_H__
#define	__PIL_RESULT_H__
#include "gafw.h"
#include <vector>
namespace GAFW
{
class Result:public FactoryAccess, public Identity, public LogFacility
{
private:
    Result(const Result& m){};
protected:
    inline Result(); //required fro ProxyResult declaration
    inline Result(Factory *f,std::string nickname,std::string name);
public:
    virtual ArrayDimensions getDimensions(CalculationId id)=0;
    virtual void getValue(CalculationId id,std::vector<unsigned int> &position, ValueWrapperBase &value)=0;
    virtual void getSimpleArray(CalculationId id, PointerWrapperBase &values)=0;
    virtual void getValues(CalculationId id,PointerWrapperBase &value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,...)=0;
    virtual void getValues(PointerWrapperBase &value, unsigned int no_of_elements, bool isColumnWise,unsigned int noOfIndexes,...)=0;
    virtual DataTypeBase getType(CalculationId id=-1)=0;
    virtual void requireResults()=0;
    virtual void doNotRequireResults()=0;
    virtual bool areResultsRequired()=0;
    virtual bool isDataValid(CalculationId id=-1)=0;
    virtual bool waitUntilDataValid(CalculationId id=-1)=0;
    
    
    virtual CalculationId calculate()=0;
    virtual void removeReusabilityOnNextUse()=0;
    virtual void reusable()=0;
    virtual void notReusable()=0;
    virtual bool isReusable()=0;
    virtual Array *getParent()=0;
    virtual void overwrite()=0;
    virtual void doNoOverwrite()=0;
    virtual bool isOverwrite()=0;
    virtual CalculationId getLastSnapshotId()=0;
   
};
inline Result::Result()
{
        throw Bug("The constructor Result::Result() is available only for programming convenience and should never be called");
        
}
inline Result::Result(Factory *f,std::string nickname,std::string name):FactoryAccess(f),Identity(nickname,name)
{
        LogFacility::init();
        FactoryAccess::init();
}

}

#endif	/* RESULTMATRIX_H */

