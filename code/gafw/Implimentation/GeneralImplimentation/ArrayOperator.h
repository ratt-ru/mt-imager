/* ArrayOperator.h:  Header file for the general implementation of the GAFW Operator 
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

#ifndef __GEN_ARRAYOPERATOR_H__
#define	__GEN_ARRAYOPERATOR_H__

#include <vector>
#include <string>
#include "Properties.h"

#include "gafw-impl.h"

namespace GAFW { namespace GeneralImplimentation
{
    
class ArrayOperator: public GAFW::ArrayOperator {
private:
    ArrayOperator(const ArrayOperator& orig){}; //copying is disallowed
    

protected:
    
    
    GAFW::Tools::CppProperties::Properties params;
    std::vector<Array *> inputs;
    std::vector <Array *> outputs;
    friend class Array;
    friend class Engine;
    ArrayOperator(Factory *f,string nickName,string name);
    virtual ~ArrayOperator();
public:
    std::vector<GAFW::GeneralImplimentation::Array *> _getInputs();
    std::vector<GAFW::GeneralImplimentation::Array *> _getOutputs();
    void clearInput(GAFW::Array * a);
    void setOutputs(std::vector<GAFW::Array *> outs);
    void setOutput(GAFW::Array *m);
    void setOutput(int outputNo,GAFW::Array *m);
    void setInput(GAFW::Array *m);
    void setInput(int inputNo,GAFW::Array *m);
    void setInputs(std::vector<GAFW::Array *> &vec);
    
    void clearInputs();
    std::vector<GAFW::Array *> getInputs();
    std::vector<GAFW::Array *> getOutputs();
    int clearOutput(GAFW::Array *m);
    int clearOutputs();
    GAFW::Array *a(const char * objectname);
    void *a(int NoOfOutputs,const char * input1,...);
    
    
    //Functions related to parameters
    inline void setParameter(std::string name,std::string value);
    inline void setParameter(std::string name, int value);
    inline void setParameter(std::string name,float value);
    inline void setParameter(std::string name,void * value);
    inline void setParameter(std::string name,std::complex<float> value);
    inline void setParameter(std::string name,bool value);
    inline void setParameter(std::string name,double value);
    inline std::string getStringParameter(std::string name);
    inline int getIntParameter(std::string name);
    inline float getFloatParameter(std::string name);
    inline void *  getPointerParameter(std::string name);
    inline bool  getBoolParameter(std::string name);
    inline double getDoubleParameter(std::string name);
    inline std::complex<float> getComplexParameter(std::string name);
    
    virtual void setParameter(ParameterWrapperBase &value);
    virtual ParameterWrapperBase getParameter(std::string name);
    virtual void getParameter(ParameterWrapperBase& saveTo);
    
    virtual inline enum GAFW::Tools::CppProperties::Properties::PropertyType getParameterType(std::string name);
    virtual inline bool isParameterSet(std::string name);
    virtual inline void deleteParameter(std::string name);
    
    void checkParameter(std::string param,enum GAFW::Tools::CppProperties::Properties::PropertyType type);
    
};
 inline void ArrayOperator::setParameter(std::string name,std::string value)
 {
     this->params.setProperty(name,value);
 }
 inline void ArrayOperator::setParameter(std::string name, int value)
 {
     this->params.setProperty(name,value);
 }
 inline void ArrayOperator::setParameter(std::string name,float value)
 {
     this->params.setProperty(name,value);
 }
 inline void ArrayOperator::setParameter(std::string name,double value)
 {
     this->params.setProperty(name,value);
 }
 
 inline void ArrayOperator::setParameter(std::string name,void * value)
 {
     this->params.setProperty(name,value);
 }
 inline void ArrayOperator::setParameter(std::string name,bool value)
 {
     this->params.setProperty(name,value);
 }
 
 inline void ArrayOperator::setParameter(std::string name,std::complex<float> value)
 {
     this->params.setProperty(name,value);
 }
/* inline std::string ArrayOperator::getParameter(std::string name)
 {
     return this->params.getProperty(name);
 }*/
 inline std::string ArrayOperator::getStringParameter(std::string name)
 {
     return this->params.getStringProperty(name);
 }
 inline int ArrayOperator::getIntParameter(std::string name)
 {
     return this->params.getIntProperty(name);
 }
 inline float ArrayOperator::getFloatParameter(std::string name)
 {
     return this->params.getFloatProperty(name);
 }
 inline double ArrayOperator::getDoubleParameter(std::string name)
 {
     return this->params.getDoubleProperty(name);
 }
 
 inline void *  ArrayOperator::getPointerParameter(std::string name)
 {
     return this->params.getPointerProperty(name);
 }
 
/* inline std::string ArrayOperator::getParameter(std::string name)
 {
     return this->params.getProperty(name);
 }
*/ 
 inline std::complex<float> ArrayOperator::getComplexParameter(std::string name)
 {
     return this->params.getComplexProperty(name);
 }
 inline bool ArrayOperator::getBoolParameter(std::string name)
 {
     return this->params.getBoolProperty(name);
 }
 
 inline enum GAFW::Tools::CppProperties::Properties::PropertyType ArrayOperator::getParameterType(std::string name)
 {
     return this->params.getPropertyType(name);
 }
 
 inline bool ArrayOperator::isParameterSet(std::string name)
 {
     return this->params.isPropertySet(name);
     
 }
 inline void ArrayOperator::deleteParameter(std::string name)
 {
     this->params.deleteProperty(name);
 }
 inline std::vector<GAFW::GeneralImplimentation::Array *> ArrayOperator::_getInputs()
 {
     return this->inputs;
 }
 inline std::vector<GAFW::GeneralImplimentation::Array *> ArrayOperator::_getOutputs()
 {
     return this->outputs;
 }














}}//end of namespace

#endif	/* MATRIXOPERATOR_H */

