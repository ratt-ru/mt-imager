/* ArrayOperator.h:  Header file for the GAFW PIL definition of an ArrayOperator.
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

#ifndef __PIL_ARRAYOPERATOR_H__
#define	__PIL_ARRAYOPERATOR_H__


#include "gafw.h"
namespace GAFW
{
    
class ArrayOperator: public FactoryAccess,public Identity, public LogFacility {
private:
    ArrayOperator(const ArrayOperator& orig){}; //copying is disallowed
protected:
    inline ArrayOperator();
    inline ArrayOperator(GAFW::Factory *f,std::string objectName,std::string name);
public:    
    virtual void clearInput(Array * a)=0;
    virtual void setOutputs(std::vector<Array *> outs)=0;
    virtual void setOutput(Array *m)=0;
    virtual void setOutput(int outputNo,Array *m)=0;
    virtual void setInput(Array *m)=0;
    virtual void setInput(int inputNo,Array *m)=0;
    virtual void setInputs(std::vector<Array *> &vec)=0;
    virtual void clearInputs()=0;
    virtual std::vector<Array *> getInputs()=0;
    virtual std::vector<Array *> getOutputs()=0;
    virtual int clearOutput(Array *a)=0;
    virtual int clearOutputs()=0;
    virtual Array *a(const char * nickname)=0;
    virtual void *a(int NoOfOutputs,const char * input1,...)=0;
    //Functions related to parameters
    virtual void setParameter(ParameterWrapperBase & value)=0;
    virtual ParameterWrapperBase getParameter(std::string name)=0;
    virtual void getParameter(ParameterWrapperBase& saveTo)=0;
    
    
    virtual bool isParameterSet(std::string name)=0;
    virtual void deleteParameter(std::string name)=0;
    
};

inline ArrayOperator::ArrayOperator(GAFW::Factory *f,std::string objectName,std::string name):Identity(objectName,name),FactoryAccess(f)
{
    FactoryAccess::init();
    LogFacility::init();
    
}
inline ArrayOperator::ArrayOperator()
{
    throw Bug("The function ArrayOperator::ArrayOperator is available only for programming convenience and should never be called");
}














}//end of namespace

#endif	/* MATRIXOPERATOR_H */

