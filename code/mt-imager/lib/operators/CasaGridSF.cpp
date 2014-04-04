/* CasaGridSF.cpp:  Pure C++ implementation of the CasaGridSF operator 
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
#include "ImagerOperators.h"
#include <string>
#include <vector>
#include "Properties.h"
using namespace mtimager;
using namespace GAFW;
using namespace GAFW::GPU; using namespace GAFW::GeneralImplimentation;
using namespace std;
using namespace GAFW::Tools::CppProperties;



CasaGridSF::CasaGridSF(GPUFactory* factory,string nickname):GPUArrayOperator(factory,nickname,"Operator::gridsf")
{
    
}

CasaGridSF::~CasaGridSF() {
}
void CasaGridSF::validate(){
    
    //TO SEEIF all parameters are required infact for all cases
    /********************************************************************
     * VAlidations:                                                     *
     * 1. No input Matrixes *
     *  Output matrix is set to real_float   
    
     * Output matrix will be set 1x400 
     * ******************************************************************/
  
   //No of inputs
    if (this->outputs.size()!=1) throw ValidationException("Only one output is supported");
    logDebug(validation,"No of outputs are correct");
    logDebug(validation,"Now validating parameters");
    this->checkParameter("support",Properties::Int);
    this->checkParameter("sampling",Properties::Int);
    this->checkParameter("lmPlanePostCorrection",Properties::Bool);
    this->checkParameter("generate",Properties::Bool);
    if (!this->isParameterSet("double-precision")) this->setParameter("double-precision",false);
    

    int support=this->params.getIntProperty("support");
    if (support<1) 
           throw ValidationException("support requested is less then 1");
    if (support<7) 
        logError(validation,"Support set to less then the minimum of 7. Convolution function is incomplete");
    if (support%2==0)
        throw ValidationException("Support is expected to be odd");
    int sampling=this->params.getIntProperty("sampling");
    if (sampling<1)
        throw ValidationException("Sampling must be set to at least 1");


    
    if (this->getBoolParameter("lmPlanePostCorrection")==false)
    {
        if (this->getBoolParameter("generate")==false) throw ValidationException("Only generate=true is supported for lmPlanePostCorrection=false");
        if (this->inputs.size()!=0) throw ValidationException("This operator does not take any inputs but validation found inputs");
        //logDebug(validation,this->outputs[0],"Defining to real_float with 400 columns and 1 row");
        if (this->isParameterSet("complexOutput")) this->checkParameter("complexOutput",Properties::Bool);
        else
            this->setParameter("complexOutput",false);
        bool complex_output=this->getBoolParameter("complexOutput");
        bool double_presicion=this->getBoolParameter("double-precision");
        
        if (complex_output&&double_presicion)  this->outputs[0]->setType(complex_double);
        else if (complex_output&&(!double_presicion)) this->outputs[0]->setType(complex_float);
        else if ((!complex_output)&&(double_presicion)) this->outputs[0]->setType(real_double);  
        else if ((!complex_output)&&(!double_presicion)) this->outputs[0]->setType(real_float);  
        
        if (sampling%2)
                this->outputs[0]->setDimensions(ArrayDimensions(2,support*sampling,support*sampling));  
        else
                this->outputs[0]->setDimensions(ArrayDimensions(2,support*sampling+1,support*sampling+1));  
    
    }
    else
    {
        if (this->getBoolParameter("generate")==false)
        {
                if (this->inputs.size()!=1) throw ValidationException("An input is expected as to correct");
                if (this->outputs.size()!=1) throw ValidationException("Only one output is supported");
                //We check dimensions of input
                if (this->inputs[0]->getDimensions().getNoOfDimensions()<2) throw ValidationException("The input is expected to be at least 2 dimensional");
                if (!this->inputs[0]->isDefined()) throw ValidationException("Input is not defined");
                if ((this->inputs[0]->getType()!=real_float)&&(this->inputs[0]->getType()!=real_double)) throw ValidationException("Input is expected to be of real_float or real_double type");
                
                // Set output
                this->outputs[0]->setDimensions(this->inputs[0]->getDimensions());
                this->outputs[0]->setType(this->inputs[0]->getType());
        }
        else
        {
                if (this->inputs.size()!=0) throw ValidationException("No input is expected");
                if (this->outputs.size()!=1) throw ValidationException("Only one output is supported");
 
                this->checkParameter("nx",Properties::Int);
                this->checkParameter("ny",Properties::Int);
                int nx,ny;
                nx=this->getIntParameter("nx");
                ny=this->getIntParameter("ny");
                if ((nx<1)||(ny<1)) throw ValidationException("Parameters nx or ny must be at least 1");
                if (this->getBoolParameter("double-precision")) this->outputs[0]->setType(real_double);
                else this->outputs[0]->setType(real_float);
                this->outputs[0]->setDimensions(ArrayDimensions(2,ny,nx));    
        }
    }
        
            
 }
   


    