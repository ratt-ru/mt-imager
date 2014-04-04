/* Array.h:  Header file for the definition of the general implementation of the Factory.
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

#ifndef __GEN_FACTORY_H__
#define	__GEN_FACTORY_H__
#include <string> 
#include <vector>
#include "gafw-impl.h"
namespace GAFW { namespace GeneralImplimentation
{
    

    class Factory:public  GAFW::Factory {
    private:
        Factory(const Factory& orig){}; //copying is not allowed
        
    protected:
        GAFW::FactoryStatisticsOutput * statisticOutput;
        
        Factory(std::string nickname, std::string name,GAFW::FactoryStatisticsOutput * statisticOutput);
        Factory();
        ~Factory();  //In the future this will delete all objects
        
        std::vector<FactoryHelper *> helpers;
        virtual GAFW::ArrayOperator *requestMyOperator(GAFW::Identity &orig,std::string &objectname, std::string &operatorName, int noOfInputs, const char *  inputObjectName,...);
        virtual GAFW::Array *requestMyArray(GAFW::Identity & origin,std::string &nickname, ArrayDimensions &dim,DataTypeBase &type);
        virtual GAFW::ArrayOperator *requestMyOperator(GAFW::Identity & origin,std::string &nickname, std::string & OperatorType);
        virtual GAFW::ArrayOperator *requestMyOperator(GAFW::Identity & origin,std::string &nickname, std::string &OperatorType, int &noOfInputs, const char *  inputNick,va_list &vars);
        virtual GAFW::ArrayOperator *requestMyOperator(GAFW::Identity & origin,std::string &nickname, std::string &OperatorType, const char * inputNick1, const char *  inputNick2, const char *  inputNick3,const char *  inputNick4,const char *  inputNick5,const char * inputNick6,const char *  inputNick7);
        virtual GAFW::ProxyResult *requestMyProxyResult(GAFW::Identity & origin,std::string &nickname,Module *mod,GAFW::Result * bind);
        virtual GAFW::Array * getMyArray(GAFW::Identity & origin,std::string &nickname);
        virtual GAFW::ArrayOperator * getMyOperator(GAFW::Identity & origin,std::string &nickname);
        virtual GAFW::ProxyResult * getMyProxyResult(GAFW::Identity & origin,std::string &nickname);
        virtual GAFW::Result *getMyResult(GAFW::Identity & origin,std::string & nickname);
        virtual Identity *getMyParent(GAFW::Identity & origin);
        
        void validateObjectName(const std::string &objectName);
        std::string getFullObjectName(GAFW::Identity &parent,string & objectname);
        virtual Result *requestResult(GAFW::Array * array);
        virtual Engine *requestEngine()=0;
        virtual DataStore *requestDataStore(DataTypeBase &type,ArrayDimensions &dim,bool allocate=true)=0;
        Registry registry;
        GAFW::ArrayOperator *createOperator(std::string objectname, std::string OperatorType);
    public:

        virtual void destroyObject(void * object);
        friend class FactoryAccess;
        friend class Result;
        friend class Array;
        virtual GAFW::Array *requestArray(std::string objectname, ArrayDimensions dim=ArrayDimensions(),DataTypeBase dataType=DataType<void>());
        virtual GAFW::ArrayOperator *requestOperator(std::string objectname, std::string OperatorType);
        virtual GAFW::ArrayOperator *requestOperator(std::string objectname, std::string OperatorType, int noOfInputs, GAFW::Array * input,...);
        virtual GAFW::ArrayOperator *requestOperator(std::string objectname, std::string OperatorType, int noOfInputs, const char *  inputNick,...);
        virtual GAFW::ArrayOperator *requestOperator(std::string objectname, std::string OperatorType, GAFW::Array * input1, GAFW::Array * input2=NULL, GAFW::Array * input3=NULL,GAFW::Array *input4=NULL,GAFW::Array *input5=NULL,GAFW::Array *input6=NULL,GAFW::Array *input7=NULL);
        virtual GAFW::ArrayOperator *requestOperator(std::string objectname, std::string OperatorType, const char * inputNick1, const char *  inputNick2=NULL, const char *  inputNick3=NULL,const char *  inputNick4=NULL,const char *  inputNick5=NULL,const char * inputNick6=NULL,const char *  inputNick7=NULL);
        virtual GAFW::ProxyResult *requestProxyResult(std::string objectname,GAFW::Module *mod=NULL,GAFW::Result * bind=NULL);
        virtual GAFW::Array * getArray(std::string objectname);
        virtual GAFW::ArrayOperator * getOperator(std::string objectname);
        virtual GAFW::ProxyResult * getProxyResult(std::string objectname);
        virtual Identity* getIdentity(std::string objectname);
        virtual Identity* getParent(std::string objectname);
        __inline__ std::string getParentObjectname(std::string &);
        virtual void registerIdentity(Identity * identity);
        virtual void registerHelper(FactoryHelper *helper);
        virtual std::string getParentNickname(std::string& nickname);


    
};
std::string Factory::getParentObjectname(std::string& objectname)
{
    return Registry::getParentObjectname(objectname);
}


}}

#endif	/* FACTORY_H */

