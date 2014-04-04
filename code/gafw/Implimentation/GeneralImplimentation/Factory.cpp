/* Factory.cpp:  general implementation of the Factory.
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

//#include "log4cxx/propertyconfigurator.h"
#include <stdarg.h>
using namespace GAFW::GeneralImplimentation;
using namespace std;
Factory::Factory(std::string objectName,std::string name,GAFW::FactoryStatisticsOutput * statisticOutput):GAFW::Factory(objectName,name)
{
    this->statisticOutput=statisticOutput;
}
Factory::Factory()
{
    throw Bug("This constructor Factory::Factory() is available for programming convenience only and should never be called");
}
Factory::~Factory()
{
    //In the future this will delete all objects
}


void Factory::destroyObject(void * object)
{
    //TODO
}
Result *Factory::requestResult(GAFW::Array *m)
{
    return new Result(dynamic_cast<GAFW::GeneralImplimentation::Factory*>(this),dynamic_cast<GAFW::GeneralImplimentation::Array*>(m));
}
GAFW::ProxyResult *Factory::requestProxyResult(std::string nickname,GAFW::Module *mod,GAFW::Result *bind)
{
   this->logDebug(builder,string("requestProxyResult():Requesting a new ProxyResult with nickname ")+nickname);
   if (mod==NULL)
       this->logWarn(builder,"requestProxyResult(): No module specified (mod=NULL)");
   if (nickname==string(""))
    {
        logError(builder,"requestProxyResult(): Empty nickname specified for request of new ProxyResult. Throwing exception");
        throw GeneralException("requestProxyResult(): Objectname for new ProxyResult is empty")
    }
   if (!this->registry.isParentRegistered(nickname)) 
   {
        logError(builder,"requestProxyResult(): Parent is not registered.Throwing exception");
        throw GeneralException(string("requestProxyResult(): Parent of new ProxyResult with nickname ")+nickname+ string( " is not registered."));
   }
   GAFW::ProxyResult * p=new ProxyResult(this,nickname,mod);
   this->logDebug(builder,p,"Created. Now setting bind(if given)");
   p->setBind(bind);
   this->logDebug(builder,p,"Registering");
   this->registry.registerIdentity(p);
     this->logDebug(builder,p,"Returning ProxyResult");
   return p;

}

GAFW::Array *Factory::requestArray(std::string nickname, ArrayDimensions dim,DataTypeBase dataType)
{
  
    this->logDebug(builder,string("requestArray():Requesting a new array with nickname ")+nickname);
    if (nickname==string(""))
    {
        logError(builder,"requestArray(): Empty nickname specified for request of new Array. Throwing exception");
        throw GeneralException("requestArray(): Objectname for new matrix is empty");
    }
    if (!this->registry.isParentRegistered(nickname)) 
   {
        logError(builder,"requestArray(): Parent is not registered.Throwing exception");
        throw GeneralException(string("requestArray(): Parent of new Array with nickname ")+nickname+ string( " is not registered."));
   }
   
    
    GAFW::Array *m=new GAFW::GeneralImplimentation::Array(this,nickname);
    this->logDebug(builder,m,"Created. Now setting dimensions and type (if given)");
    m->setDimensions(dim);
    m->setType(dataType);
    this->logDebug(builder,m,"Registering");
    this->registry.registerIdentity(m);
    this->logDebug(builder,m,"Returning Array");
    return m;
}

GAFW::ArrayOperator *Factory::requestOperator(std::string nickname, std::string operatorType)
{
    this->logDebug(builder,string("requestOperator():Requesting a new operator with nickname ")+nickname);
    if (nickname==string(""))
    {
        logError(builder,"requestOperator(): Empty nickname specified for request of new Operator. Throwing exception");
        throw GeneralException("requestOperator(): Objectname for new Operator is empty")
    }
     if (!this->registry.isParentRegistered(nickname)) 
   {
        logError(builder,"requestOperator(): Parent is not registered.Throwing exception");
        throw GeneralException(string("requestOperator(): Parent of new Array with nickname ")+nickname+ string( " is not registered."));
   }
   
    GAFW::ArrayOperator *op=this->createOperator(nickname,operatorType);
    this->logDebug(builder,op,"Created. Now Registering.");    
    this->registry.registerIdentity(op);
    return op;
}

 GAFW::ArrayOperator *Factory::requestOperator(std::string nickname, std::string operatorName,int noOfInputs,GAFW::Array * input,...)
{
    GAFW::ArrayOperator *op=this->requestOperator(nickname,operatorName);
    va_list arguments;
    va_start(arguments,input);
    int y;
    for (y=0;y<noOfInputs;y++)
    {
        Array* current_input=(y==0)?dynamic_cast<GAFW::GeneralImplimentation::Array*> (input):va_arg(arguments,Array*);
        if (current_input==NULL) break;
        op->setInput(current_input);
    }
    va_end(arguments);
    stringstream ss;
    ss <<  y << " inputs found";
    this->logDebug(builder,op,string(ss.str()));
    this->logDebug(builder,op,"Returning Operator");
    return op;
    

}

GAFW::ArrayOperator *Factory::requestOperator(std::string nickname, std::string operatorName,int noOfInputs,const char * inputObjectname,...)
{
    GAFW::ArrayOperator *op=this->requestOperator(nickname,operatorName);
    va_list arguments;
    va_start(arguments,inputObjectname);
    int y;
    for (y=0;y<noOfInputs;y++)
    {
        const char * current_input_nickname=(y==0)?inputObjectname:va_arg(arguments,const char *);
        if ((current_input_nickname==NULL)||(string(current_input_nickname)==string(""))) break;
        GAFW::Array * current_input=this->getArray(current_input_nickname);
        if (current_input==NULL) break;
        op->setInput(current_input);
    }
    va_end(arguments);
    stringstream ss;
    ss <<  y << " inputs found";
    this->logDebug(builder,op,string(ss.str()));
    this->logDebug(builder,op,"Returning Operator");
    return op;
}

GAFW::ArrayOperator *Factory::requestOperator(std::string nickname, std::string operatorName, GAFW::Array * input1, GAFW::Array * input2, GAFW::Array * input3,GAFW::Array *input4,GAFW::Array* input5,GAFW::Array *input6,GAFW::Array * input7)
{
    return requestOperator(nickname,operatorName,7,input1,input2,input3,input4,input5,input6,input7);
    
}
    
GAFW::ArrayOperator *Factory::requestOperator(std::string nickname, std::string operatorName, const char *  inputNick1, const char *  inputNick2, const char *  inputNick3,const char *  inputNick4,const char *  inputNick5,const char * inputNick6,const char *  inputNick7)
{
    return requestOperator(nickname,operatorName,7,inputNick1,inputNick2,inputNick3,inputNick4,inputNick5,inputNick6,inputNick7);
}
GAFW::Array * Factory::getArray(std::string nickname)
{
    
    //Most logs in below function
    Identity * ob=this->registry.getIdentity(nickname);
    Array * myArray=dynamic_cast<Array *>(ob);
    if (myArray==NULL) throw GeneralException("Getting an object which is not an array");
    return myArray;
    
}
GAFW::ArrayOperator * Factory::getOperator(std::string nickname)
{
    
    //All logs in below function
    Identity * ob= this->registry.getIdentity(nickname);
    GAFW::ArrayOperator * myOperator=dynamic_cast<ArrayOperator *>(ob);
    if (myOperator==NULL) throw GeneralException("Getting an object which is not an array");
    return myOperator;
    
}
GAFW::ProxyResult * Factory::getProxyResult(std::string nickname)
{
    
    //All logs in below function
    Identity * ob= this->registry.getIdentity(nickname);
    GAFW::ProxyResult * myProxyResult=dynamic_cast<ProxyResult *>(ob);
    return myProxyResult;
}

void Factory::registerIdentity(Identity * identity)
{
    this->registry.registerIdentity(identity);
}
GAFW::ArrayOperator * Factory::createOperator(std::string nickname, std::string operatorName)
{
    GAFW::ArrayOperator *ret;
    for (vector<FactoryHelper*>::reverse_iterator i=this->helpers.rbegin();i<this->helpers.rend();i++)
    {
        logDebug(builder,string("Checking with HelperFactory named ")+(*i)->getName());
        ret=(*i)->createOperator(nickname,operatorName);
        if (ret!=NULL) return ret;
    }
    
    string s="Unknown or unsupported operator: ";
    s+=operatorName;
    throw GeneralException(s);
}
void Factory::registerHelper(FactoryHelper * helper)
{
    
    this->helpers.push_back(helper->reCreateForFactory(this));
}
Identity * Factory::getParent(std::string nickname)
{
    return this->registry.getParent(nickname);
}

void Factory::validateObjectName( const std::string& objectname)
{
    if (objectname==string(""))
    {
        throw GeneralException("Objectname empty");
    }
}

std::string Factory::getFullObjectName(GAFW::Identity &parent,string & objectname)
{
    if (parent.getObjectName()==string(""))
    {
        return objectname;
    }
    return parent.getObjectName()+"."+objectname;
}


GAFW::Array *Factory::requestMyArray(GAFW::Identity &orig,std::string &objectname, ArrayDimensions &dim,DataTypeBase &type)
{
    //First thing ..check that objectname is not empty and that and I don't have such child
    this->validateObjectName(orig.objectName);
    std:string completeNick=this->getFullObjectName(orig,objectname);
    GAFW::Array * a=this->requestArray(completeNick,dim,type);
    return a;
}

GAFW::ArrayOperator *Factory::requestMyOperator(GAFW::Identity &orig,std::string &objectname, std::string &OperatorType)
{
    this->validateObjectName(orig.objectName);
    std:string completeNick=this->getFullObjectName(orig,objectname);
    GAFW::ArrayOperator * a=this->requestOperator(completeNick,OperatorType);
    return a;
}
GAFW::ArrayOperator *Factory::requestMyOperator(GAFW::Identity &orig,std::string &objectname, std::string &operatorName, int noOfInputs, const char *  inputObjectName,...)
{
    va_list arguments;
    va_start(arguments,inputObjectName);
    GAFW::ArrayOperator *op=this->requestMyOperator(orig,objectname,operatorName,noOfInputs,inputObjectName,arguments);
    va_end(arguments);
    return op;
}
GAFW::ArrayOperator *Factory::requestMyOperator(GAFW::Identity & orig,std::string &objectname, std::string &OperatorType, int &noOfInputs, const char *  inputObjectname,va_list &vars)
{
    GAFW::ArrayOperator *op=this->requestMyOperator(orig,objectname,OperatorType);
    int y;
    for (y=0;y<noOfInputs;y++)
    {
        const char * current_input_objectname=(y==0)?inputObjectname:va_arg(vars,const char *);
        if (current_input_objectname==NULL) break;
        string current_input_objectname_s=string(current_input_objectname);
        if (current_input_objectname_s==string("")) break;
        
        GAFW::Array * current_input=this->getMyArray(orig,current_input_objectname_s);
        if (current_input==NULL) break;
        op->setInput(current_input);
    }
    stringstream ss;
    ss <<  y << " inputs found";
    this->logDebug(builder,op,string(ss.str()));
    this->logDebug(builder,op,"Returning Operator");
    return op;
    
}
GAFW::ArrayOperator *Factory::requestMyOperator(GAFW::Identity &orig,std::string &nickname, std::string &OperatorType, const char * inputNick1, const char *  inputNick2, const char *  inputNick3,const char *  inputNick4,const char *  inputNick5,const char * inputNick6,const char *  inputNick7)
{
    return this->requestMyOperator(orig,nickname,OperatorType,7, inputNick1, inputNick2, inputNick3, inputNick4,inputNick5, inputNick6,inputNick7);
}   
GAFW::ProxyResult *Factory::requestMyProxyResult(GAFW::Identity &orig,std::string &objectname,Module *mod,GAFW::Result * bind)
{
    std:string completeNick=this->getFullObjectName(orig,objectname);
    GAFW::ProxyResult * a=this->requestProxyResult(completeNick,mod,bind);
    return a;
    
}
GAFW::Array * Factory::getMyArray(GAFW::Identity &orig,std::string &objectname)
{
    std:string completeObjectName=this->getFullObjectName(orig, objectname);
    return this->getArray(completeObjectName);
    
}
GAFW::ArrayOperator * Factory::getMyOperator(GAFW::Identity &orig,std::string &objectname)
{
    std:string completeObjectName=this->getFullObjectName(orig, objectname);
    return this->getOperator(completeObjectName);
      
}
GAFW::ProxyResult * Factory::getMyProxyResult(GAFW::Identity &orig,std::string& objectname)
{
    std:string completeObjectName=this->getFullObjectName(orig, objectname);
    return this->getProxyResult(completeObjectName);
    
}
GAFW::Result *Factory::getMyResult(GAFW::Identity &orig,std::string &objectname)
{
    std:string completeObjectName=this->getFullObjectName(orig, objectname);
    return this->getArray(completeObjectName)->getResults();
}

Identity *Factory::getMyParent(GAFW::Identity &orig)
{
    return this->getParent(orig.objectName);
}
 std::string Factory::getParentNickname(std::string& nickname)
{
    return Registry::getParentObjectname(nickname);
}
 Identity* Factory::getIdentity(std::string objectname)
 {
     return this->registry.getIdentity(objectname);
 }
        