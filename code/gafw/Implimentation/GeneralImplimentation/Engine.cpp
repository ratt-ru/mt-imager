/* Engine.cpp:  General implementation of the Engine.
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
#include <iostream>
using namespace GAFW::GeneralImplimentation;
using namespace std;
    
Engine::Engine(Factory * factory,std::string nickname,std::string name):FactoryAccess(factory),Identity(nickname,name) {
    
    FactoryAccess::init();
    LogFacility::init();
   this->logDebug(other,"Engine() constructor called");
}
Engine::Engine()
{
    throw Bug("The constructor Engine::Engine() is available for programming convenience only and should never be called");
}



Engine::~Engine() {
    this->logDebug(other,"~Engine() Destructor called");
}
void Engine::getArrayInternalData(Array *m, ArrayInternalData &d)
{
   // d.factory=dynamic_cast<Factory *>( m->factory);
    d.input_to=m->input_to;
    d.output_of=m->output_of;
    d.result=m->result;
    d.result_Outputof=m->result_Outputof;
    d.store=m->store;
    d.dim=(*(m->tmp_dim));
    d.type=(StoreType)m->type._type;
    d.preValidator=m->preValidator;
    d.preValidatorDependents=m->preValidatorDependents;
    d.toReuse=d.result->toReuse;
    d.toOverwrite=d.result->toOverwrite;
    d.requireResult=d.result->resultsRequired;
    d.removeReusability=d.result->removeReusability;
}

DataStore * Engine::getResultStore(Result *r,int id)
{
    return r->getStore(id);
}
void Engine::createResultStore(Result *r,int id,StoreType type, ArrayDimensions &dim)
{
        
        switch (type)
        {
            case complex_float:
            {
                DataType<std::complex<float> > d;
                r->createStore(id,d,dim);
            }
                break;
            case complex_double:
            {
                DataType<std::complex<double> > d;
                r->createStore(id,d,dim);
            }
                break;
            case real_float:
            {
                DataType<float > d;
                r->createStore(id,d,dim);
            }
                break;
            case real_double:
            {
                DataType<double > d;
                r->createStore(id,d,dim);
            }
                break;
            case real_int:
            {
                DataType<int > d;
                r->createStore(id,d,dim);
            }
                break;
            
            case real_uint:
            {
                DataType<unsigned int> d;
                r->createStore(id,d,dim);
            }
                break;
            
            case real_longint:
            {
                DataType<long int> d;
                r->createStore(id,d,dim);
            }
                break;
            
            case real_shortint:
            {
                DataType<short int> d;
                r->createStore(id,d,dim);
            }
                break;
            
            case real_ulongint:
            {
                DataType<unsigned long int> d;
                r->createStore(id,d,dim);
            }
                break;
            
            case real_ushortint:
            {
                DataType<unsigned short int> d;
                r->createStore(id,d,dim);
            }
                break;
            
            default:
                throw GeneralException("BUG: unkw store type to create");
                
        }
        
    
    
}    
bool Engine::areResultsRequired(Result *r)
{
    return r->areResultsRequired();
}
bool Engine::areResultsToBeUsed(Result *r)
{
    return r->isReusable();
}
bool Engine::areResultsToOverwrite(Result* r)
{
    return r->toOverwrite;
}
bool Engine::isCopyResult(Result *r)
{
    return dynamic_cast<GAFW::ProxyResult*>(r)?true:false;
}
GAFW::Result * Engine::findProperResult(GAFW::Result *res, vector<ProxyResult*> &chain)
{
    GAFW::Result *ans=res;
    while (dynamic_cast<GAFW::ProxyResult*>(ans)?true:false)
    {
        chain.push_back(dynamic_cast<ProxyResult *>(ans));
        ans=dynamic_cast<GAFW::Result *>((dynamic_cast<ProxyResult *>(ans))->getBind());
        if (ans==NULL) throw ValidationException("ProxyResult object is not bound");
        res=ans;
    }
    return ans;
}
GAFW::Result * Engine::findProperResult(GAFW::Result *res)
{
    GAFW::Result *ans=res;
    while (dynamic_cast<GAFW::ProxyResult*>(ans)?true:false)
    {
       
        ans=dynamic_cast<Result *>(dynamic_cast<ProxyResult *>(ans)->getBind());
        if (ans==NULL) throw ValidationException("ProxyResult object is not bound");
        res=ans;
    }
    if (dynamic_cast<GAFW::GeneralImplimentation::Array*>(ans->getParent())->result_Outputof!=NULL) return this->findProperResult(dynamic_cast<GAFW::GeneralImplimentation::Array*>(ans->getParent())->result_Outputof);
    return ans;
}
GAFW::Tools::CppProperties::Properties& Engine::getOperatorParamsObject(ArrayOperator * oper)
{
    return oper->params;
}