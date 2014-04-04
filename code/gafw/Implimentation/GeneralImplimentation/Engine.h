/* Array.h:  Header file for the definition of the Engine.
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


#ifndef __GEN_ENGINE_H__
#define	__GEN_ENGINE_H__
#include <vector>
namespace GAFW { namespace GeneralImplimentation {
 
class Engine: public FactoryAccess, public Identity ,public LogFacility {
private:
    Engine(const Engine& orig){};
    
protected:
    Engine(Factory *factory,std::string nickname,std::string name);
    Engine();
    
    ~Engine();
    void getArrayInternalData(Array *m, ArrayInternalData &d);
    DataStore * getResultStore(Result *r,int id);
//    void setResultStatus(Result *r,int id, ResultStatus status);
    void createResultStore(Result *r,int id,StoreType type, ArrayDimensions &dim);
    bool areResultsRequired(Result *r);  //Available for completness as function is publicly available
    bool areResultsToBeUsed(Result *r);
    bool areResultsToOverwrite(Result *r);
    //int getResultLastSnapshotId(Result *r);
    bool isCopyResult(Result *r);
    GAFW::Result * findProperResult(GAFW::Result *r, std::vector<ProxyResult*> &chain);
    GAFW::Result * findProperResult(GAFW::Result *r);
    GAFW::Tools::CppProperties::Properties& getOperatorParamsObject(ArrayOperator * oper);
    
public:
    virtual long long int  calculate(Result *r)=0;
    
};
} }

#endif	/* ENGINE_H */

