/* Factory.h:  Definition of the Factory .     
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

#ifndef __PIL_FACTORY_H__
#define	__PIL_FACTORY_H__
#include <string> 
#include <vector>
#include <stdarg.h>
#include "gafw.h"
namespace GAFW {
    

class Factory:public LogFacility,public Identity {
protected:
    inline Factory(std::string objectName,std::string name);
    inline Factory();
    //All below function are required for direct access from FactoryAccess
    virtual Array *requestMyArray(Identity & origin,std::string &nickname, ArrayDimensions &dim,DataTypeBase &type)=0;
    virtual ArrayOperator *requestMyOperator(Identity & origin,std::string &nickname, std::string & OperatorType)=0;
    virtual  ArrayOperator *requestMyOperator(Identity & origin,std::string &nickname, std::string &OperatorType, int &noOfInputs, const char *  inputNick,va_list &vars)=0;
    virtual ArrayOperator *requestMyOperator(Identity & origin,std::string &nickname, std::string &OperatorType, const char * inputNick1, const char *  inputNick2, const char *  inputNick3,const char *  inputNick4,const char *  inputNick5,const char * inputNick6,const char *  inputNick7)=0;
    virtual ProxyResult *requestMyProxyResult(Identity & origin,std::string &nickname,Module *mod,Result * bind)=0;
    virtual Array * getMyArray(Identity & origin,std::string &nickname)=0;
    virtual ArrayOperator * getMyOperator(Identity & origin,std::string &nickname)=0;
    virtual ProxyResult * getMyProxyResult(Identity & origin,std::string &nickname)=0;
    virtual Result *getMyResult(Identity & origin,std::string & nickname)=0;
    virtual Identity *getMyParent(Identity & origin)=0;
    friend class FactoryAccess;
public:
    virtual void destroyObject(void * object)=0;
    virtual Array *requestArray(std::string nickname, ArrayDimensions dim=ArrayDimensions(),DataTypeBase dataType=DataType<void>())=0;
    virtual ArrayOperator *requestOperator(std::string nickname, std::string OperatorType)=0;
    virtual ArrayOperator *requestOperator(std::string nickname, std::string OperatorType, int noOfInputs, Array * input,...)=0;
    virtual ArrayOperator *requestOperator(std::string nickname, std::string OperatorType, int noOfInputs, const char *  inputNick,...)=0;
    virtual ArrayOperator *requestOperator(std::string nickname, std::string OperatorType, Array * input1, Array * input2=NULL, Array * input3=NULL,Array *input4=NULL,Array *input5=NULL,Array *input6=NULL,Array *input7=NULL)=0;
    virtual ArrayOperator *requestOperator(std::string nickname, std::string OperatorType, const char * inputNick1, const char *  inputNick2=NULL, const char *  inputNick3=NULL,const char *  inputNick4=NULL,const char *  inputNick5=NULL,const char * inputNick6=NULL,const char *  inputNick7=NULL)=0;
    virtual ProxyResult *requestProxyResult(std::string nickname,Module *mod=NULL,Result * bind=NULL)=0;
    virtual Array * getArray(std::string nickname)=0;
    virtual ArrayOperator * getOperator(std::string nickname)=0;
    virtual ProxyResult * getProxyResult(std::string nickname)=0;
    virtual Identity* getIdentity(std::string nickname)=0;
    virtual Identity* getParent(std::string nickname)=0;
    virtual std::string getParentNickname(std::string &)=0;
    virtual void registerIdentity(Identity * identity)=0;
    virtual void registerHelper(FactoryHelper *helper)=0;
    
};
inline Factory::Factory(std::string objectName,std::string name):Identity(objectName,name)
{
    LogFacility::init();
}
inline Factory::Factory()
{
    throw Bug("The constructor Factory::Factory() is a available only for programming convenience and should never be called");
}

}

#endif	/* FACTORY_H */

