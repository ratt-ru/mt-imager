/* ArrayOperator.cpp:  General Implementation of GAFW Operator.
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

#include "ArrayOperator.h"
#include <stdarg.h>
#include <complex>
using namespace GAFW::GeneralImplimentation;
using namespace std;
ArrayOperator::ArrayOperator(Factory *f,string nickName,string name):GAFW::ArrayOperator(f,nickName,name)
{
    this->inputs.clear();
    this->outputs.clear();
}


ArrayOperator::~ArrayOperator() 
{

}

void ArrayOperator::setOutput(GAFW::Array *a)
{
    if (a==NULL) throw GeneralException("Input is NULL");
    //Go through the outputs vector and ensure that the item is not already there 
    //as it does not make any sense to have doubles
    for (vector<Array *>::iterator i=this->outputs.begin();i<this->outputs.end();i++)
    {
        if (*i==a) throw GeneralException("Array is already set as output");
    }
    this->outputs.push_back(dynamic_cast<Array *>(a));
    dynamic_cast<Array *>(a)->output_of=this;
    
    //HERE THERE IS A BUG THAT NEEDS SOLUTION... what if output was already an output  somewhere else 
}
void ArrayOperator::setOutput(int outputNo,GAFW::Array *a)
{
    if (a==NULL) throw GeneralException("Input is NULL");
    //Go through the outputs vector and ensure that the item is not already there 
    //as it does not make any sense to have doubles
    for (vector<Array *>::iterator i=this->outputs.begin();i<this->outputs.end();i++)
    {
        if (*i==a) throw GeneralException("Array is already set as output");
    }
    if (outputNo<this->outputs.size())
    {
        //this->outputs[outputNo]->output_of=NULL; ///This code is BUGGYIy breaks code also... for arrays that contain results but are not set as output they are taken as input
        this->outputs[outputNo]=dynamic_cast<Array *>(a);
        dynamic_cast<Array *>(a)->output_of=dynamic_cast<ArrayOperator *>(this);
    }
    else if (outputNo==this->outputs.size())
    {
        this->outputs.push_back(dynamic_cast<Array *>(a));
        dynamic_cast<Array *>(a)->output_of=dynamic_cast<ArrayOperator *>(this);
    }
    else
        throw GeneralException("Output Index is greater then outputs stored");
}

void ArrayOperator::setOutputs(vector<GAFW::Array *> outs )
{
    for (vector<GAFW::Array *>::iterator i=outs.begin();i<outs.end();i++)
    {
        if (*i==NULL) throw GeneralException("One of the Arrays inputted to set as output is NULL");
        for (vector<Array *>::iterator j=this->outputs.begin();j<this->outputs.end();j++)
        {
        if (*j==*i) throw GeneralException("One of the input Arrays is already set as output");
        }
        //Check for double entries sin outs
        for (vector<GAFW::Array *>::iterator j=outs.begin();j<i;j++)
        {
        if (*j==*i) throw GeneralException("Found double entries in inputted Array vector");
        }
        
    }
    for (vector<GAFW::Array *>::iterator i=outs.begin();i<outs.end();i++)
    {
        this->outputs.push_back(dynamic_cast<Array*>(*i));
        dynamic_cast<Array *>(*i)->output_of=dynamic_cast<ArrayOperator *>(this);
    }
}

void ArrayOperator::setInput(GAFW::Array *a)
{
    //A ArrayOpeartor can have many inputs and can be duplicate..
    //Once we insert it here we need to also insert ourselves in the matrix vector
    if (a==NULL)  throw GeneralException("Input is NULL");
    this->inputs.push_back(dynamic_cast<Array *>(a));
    dynamic_cast<Array *>(a)->input_to.push_back(this);
    
    
}
 void ArrayOperator::setInput(int inputNo,GAFW::Array *a)
 {
     if (a==NULL) throw GeneralException("Trying to set input to NULL");
     this->inputs[inputNo]=dynamic_cast<Array*>(a);
 }
void ArrayOperator::setInputs(vector<GAFW::Array *> & vec)
{
    //same arguments as setInput(Array *m) .. we will call the function several times.. If a NULL is
    //encountered then setInput(Array *m) will throw an exception for us
    for (vector<GAFW::Array*>::iterator i=vec.begin();i<vec.end();i++)
        this->setInput(dynamic_cast<Array*>(*i));  //Note that *i is a pointer
    
}
void ArrayOperator::clearInputs()
{
    throw GeneralException("Function not yet implemented");
}
void ArrayOperator::clearInput(GAFW::Array * a)
{
    // We  clear a particular input.. Note that in case the input is repeated 
    //more then once then we only remove the last one
    bool found=false;
    if (a==NULL) throw GeneralException ("Requested to clear a NULL input");
    //Search for the element
    for (vector<Array *>::reverse_iterator r=this->inputs.rbegin();r<this->inputs.rend();r++)
    {
        if ((*r)==a) 
        {    //this->inputs.erase(r); //to see
             found=true;
             break;           
        }
    }
    if (!found) throw GeneralException("Input Array was not found");
    found=false;
    for (vector<ArrayOperator *>::reverse_iterator r=dynamic_cast<Array *>(a)->input_to.rbegin();r< dynamic_cast<Array *>(a)->input_to.rend();r++)
    {
        if ((*r)==this) 
        {   // m->input_to.erase(r); //to see
             found=true;
             break;           
        }
    }
    if (!found) throw GeneralException("Input array was not found in the Array input_to vector. This is a bug whereby the vectors are not in sync"); 
    throw GeneralException("Function is not yet ready")

}


vector<GAFW::Array *> ArrayOperator::getInputs()
{
    throw GeneralException("Function not yet implemented");
}
vector <GAFW::Array *> ArrayOperator::getOutputs()
{
    throw GeneralException("Function not yet implemented");
    
}
int ArrayOperator::clearOutput(GAFW::Array* m)
{
    throw GeneralException("Function not yet implemented");
}
int ArrayOperator::clearOutputs()
{
    this->outputs.clear(); //This is not good but for now is OK
   // throw GeneralException("Function not yet implemented");
}
GAFW::Array * ArrayOperator::a(const char * nickname)
{
        
     GAFW::Array *a;
     if (this->getMyParent()!=NULL)
        a=this->getFactory()->requestArray(this->getMyParent()->objectName+"."+nickname);
    else
        a=this->getFactory()->requestArray(nickname);
    
          
     this->logDebug(builder,a,"Setting as output array");
     this->setOutput(a);
     return a;
}
void * ArrayOperator::a(int NoOfOutputs, const char * nickname1,...)
{
    va_list arguments;
    this->a(nickname1);
    va_start(arguments,nickname1);
    int y;
    for (y=1;y<NoOfOutputs;y++)
    {
        this->a(va_arg(arguments,const char*));
    }    
    va_end(arguments);    
     
}

void ArrayOperator::checkParameter(std::string param,enum GAFW::Tools::CppProperties::Properties::PropertyType type)
{
    if (!this->params.isPropertySet(param)) throw ValidationException("Parameter "+param+ " is not set");
    if (this->params.getPropertyType(param)!=type) 
    {
        string stype;
        switch(type)
        {
            case GAFW::Tools::CppProperties::Properties::Complex:
                stype="complex";
                break;
            case GAFW::Tools::CppProperties::Properties::Float:
                stype="float";
                break;
            case GAFW::Tools::CppProperties::Properties::Int:
                stype="integer";
                break;
            case GAFW::Tools::CppProperties::Properties::Pointer:
                stype="pointer";
                break;
            case GAFW::Tools::CppProperties::Properties::String:
                stype="string";
                break;
            case GAFW::Tools::CppProperties::Properties::Bool:
                stype="boolean";
                break;
            case GAFW::Tools::CppProperties::Properties::Double:
                stype="double";
                break;
            default:
                stype="Unknown!!";
                
        }
            
        throw ValidationException("Parameter "+param+ " is expected to be of type " +stype);
    }
    
    
}
void ArrayOperator::setParameter(GAFW::ParameterWrapperBase &value)
{
    switch ((GAFW::Tools::CppProperties::Properties::PropertyType)value.parameterType)
    {
       case GAFW::Tools::CppProperties::Properties::Complex:
            this->setParameter(value.parameterName,(dynamic_cast< ParameterWrapper<std::complex<float> > &>(value)).value);
            break;
        
        case GAFW::Tools::CppProperties::Properties::Float:
            this->setParameter(value.parameterName,(dynamic_cast< ParameterWrapper<float > & >(value)).value);
 
            break;
        
        case GAFW::Tools::CppProperties::Properties::Int:
           this->setParameter(value.parameterName,dynamic_cast< ParameterWrapper<int> &>(value).value);
 
            break;
        case GAFW::Tools::CppProperties::Properties::Pointer:
           this->setParameter(value.parameterName,dynamic_cast< ParameterWrapper<void * > &>(value).value);
 
            break;

        case GAFW::Tools::CppProperties::Properties::String:
           this->setParameter(value.parameterName,dynamic_cast< ParameterWrapper<std::string> &>(value).value);
 
            break;
        case GAFW::Tools::CppProperties::Properties::Bool:
           this->setParameter(value.parameterName,dynamic_cast< ParameterWrapper<bool > &>(value).value);
 
            break;
        case GAFW::Tools::CppProperties::Properties::Double:
           this->setParameter(value.parameterName,dynamic_cast< ParameterWrapper<double > &>(value).value);
            break;
        default:
            throw Bug("Unkonwn conversion");
         
        }
}
GAFW::ParameterWrapperBase ArrayOperator::getParameter(std::string name)
{
    throw GeneralException ("Unimplimented!!");
}

void ArrayOperator::getParameter(GAFW::ParameterWrapperBase& saveTo)
{
    throw GeneralException ("Unimplimented!!");
}
    
