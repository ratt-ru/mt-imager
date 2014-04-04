/* GPUFactory.h:  Definition of the GPUFactory class. 
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

#ifndef __GPUMATRIXFACTORY_H__
#define	__GPUMATRIXFACTORY_H__

#include <string>

#include "GPUEngine.h"
namespace GAFW { namespace GPU
{

class GPUFactory: public  GAFW::GeneralImplimentation::Factory {
private:
    GPUFactory (const GPUFactory& f){};
protected:
/*    //This is something fishy
   using GAFW::GeneralImplimentation::Factory::requestMyArray;
   //virtual GAFW::Array *requestMyArray(FactoryAccess & origin,std::string &nickname, ArrayDimensions &dim,DataTypeBase &type);
   using GAFW::GeneralImplimentation::Factory::requestMyOperator;
   //virtual ArrayOperator *requestMyOperator(FactoryAccess & origin,std::string &nickname, std::string & OperatorType);
   //virtual  ArrayOperator *requestMyOperator(FactoryAccess & origin,std::string &nickname, std::string &OperatorType, int &noOfInputs, const char *  inputNick,va_list &vars);
   //virtual ArrayOperator *requestMyOperator(FactoryAccess & origin,std::string &nickname, std::string &OperatorType, const char * inputNick1, const char *  inputNick2, const char *  inputNick3,const char *  inputNick4,const char *  inputNick5,const char * inputNick6,const char *  inputNick7);
   using GAFW::GeneralImplimentation::Factory::getMyOperator;
   //virtual ArrayOperator * getMyOperator(FactoryAccess & origin,std::string &nickname);
   using GAFW::GeneralImplimentation::Factory::requestMyProxyResult;
   //virtual ProxyResult *requestMyProxyResult(FactoryAccess & origin,std::string &nickname,Module *mod,Result * bind);
   using GAFW::GeneralImplimentation::Factory::getMyProxyResult;
   //virtual ProxyResult * getMyProxyResult(FactoryAccess & origin,std::string &nickname);
   
   using GAFW::GeneralImplimentation::Factory::getMyArray;
   //virtual Array * getMyArray(FactoryAccess & origin,std::string &nickname);
   using GAFW::GeneralImplimentation::Factory::getMyResult;
   //virtual Result *getMyResult(FactoryAccess & origin,std::string & nickname);
   using GAFW::GeneralImplimentation::Factory::getMyParent;
   //virtual Identity *getMyParent(FactoryAccess & origin);
   
  */ 
    GPUEngine * engine;
    virtual GAFW::GPU::GPUEngine *requestEngine(); 
    virtual GAFW::GeneralImplimentation::DataStore *requestDataStore(DataTypeBase &type, GAFW::ArrayDimensions &dim,bool allocate=true);
public:
    /*virtual ArrayOperator *requestOperator(std::string nickname, std::string OperatorType, int noOfInputs, Array * input,...);
    virtual ArrayOperator *requestOperator(std::string nickname, std::string OperatorType, Array * input1, Array * input2=NULL, Array * input3=NULL,Array *input4=NULL,Array *input5=NULL,Array *input6=NULL,Array *input7=NULL);
     virtual std::string getParentNickname(std::string &);*/
    GPUFactory(GAFW::FactoryStatisticsOutput * statisticOutput);
    ~GPUFactory();

};
} }
#endif
