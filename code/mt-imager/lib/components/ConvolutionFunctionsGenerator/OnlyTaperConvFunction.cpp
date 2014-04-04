/* OnlyTaperConvFunctuion.h:  Implementation of the OnlyTaperConvFunction GAFW module. 
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

#include "OnlyTaperConvFunction.h"
#include "MTImagerException.h"
#include <sstream>
#include <iostream>
using namespace GAFW;
using namespace std;
using namespace mtimager;
using namespace GAFW::Tools::CppProperties;

OnlyTaperConvFunction::OnlyTaperConvFunction(GAFW::Factory *factory,std::string nickname,  Conf conf):Module(factory,nickname,"W-Projection Imager"),conf(conf)
{
    logDebug(other,"Initialising...");
   
    logInfo(other, "Using normal Interferometric gridding");
   
  
    this->requestMyOperator("taper",this->conf.taper_operator,"")->a("UnorderedConvFunction");
    ParameterWrapper<int> int_parameter;
    
    this->getMyOperator("taper")->setParameter(int_parameter.setNameAndValue("support",this->conf.taper_support));
    //this->getMyOperator("taper")->setParameter("support",ParameterWrapper<int>(this->conf.taper_support));
    this->getMyOperator("taper")->setParameter(int_parameter.setNameAndValue("sampling",this->conf.taper_sampling));
    ParameterWrapper<bool> bool_parameter;
    this->getMyOperator("taper")->setParameter(bool_parameter.setNameAndValue("lmPlanePostCorrection",false));
    this->getMyOperator("taper")->setParameter(bool_parameter.setNameAndValue("generate",true));
    this->getMyOperator("taper")->setParameter(bool_parameter.setNameAndValue("complexOutput",true));
    this->requestMyOperator("convFuncOrganise","ConvFunctionReOrganize","UnorderedConvFunction")->a("FinalConvFunction");
    
    
    this->getMyOperator("convFuncOrganise")->setParameter(int_parameter.setNameAndValue("ConvolutionFunction.sampling",this->conf.taper_sampling));
    this->getMyOperator("convFuncOrganise")->setParameter(int_parameter.setNameAndValue("ConvolutionFunction.support",this->conf.taper_support));
    this->requestMyArray("convInfo",ArrayDimensions(2,1,2),DataType<int>());
    
    this->requestMyOperator("SumConvolutionFunctions","SumConvolutionFunctions","convInfo","FinalConvFunction")->a("ConvSums");
    this->getMyOperator("SumConvolutionFunctions")->setParameter(int_parameter.setNameAndValue("sampling",this->conf.taper_sampling));
    this->getMyArray("ConvSums")->getResults()->reusable();
    this->getMyArray("FinalConvFunction")->getResults()->reusable();
    
    logDebug(other,"Initialisation complete");
}
OnlyTaperConvFunction::~OnlyTaperConvFunction()
{
    
}
/*
void OnlyTaperConvFunction::paramLoader(CppProperties::Properties& oparams)
{
    logDebug(other,"Loading Parameters...");
    logWarn(other,"paramLoader() will be changed in the future");
    this->params.taper_operator=oparams.getStringProperty("taper.operator");
    this->params.taper_support=oparams.getIntProperty("taper.support");
    this->params.taper_sampling=oparams.getIntProperty("taper.sampling");
    this->params.conv_function_support=oparams.getIntProperty("wproj.convFunction.support");
    this->params.conv_function_sampling=oparams.getIntProperty("wproj.convFunction.sampling");
    this->params.wplanes=oparams.getIntProperty("wproj.wplanes");    
   
}*/
void OnlyTaperConvFunction::reset()
{
    
}
void OnlyTaperConvFunction::calculateConvFunction()
{
    this->calculate();
}

void OnlyTaperConvFunction::calculate()
{
    ValueWrapper<int> int_value;
    vector<unsigned int> p;
    p.push_back(0);
    p.push_back(0);
    Array * convInfo=this->getMyArray("convInfo");
    convInfo->setValue(p,int_value.setValue(this->conf.taper_support));
    p[1]=1;
    convInfo->setValue(p,int_value.setValue(0));
    //this->getFactory()->getArray("ConvSums")->getResults()->calculate();
    this->getMyResult("ConvSums")->calculate();

}
void OnlyTaperConvFunction::setInput(int inputNo, GAFW::Result *res)
{
    throw ImagerException("No inputs expected");
    
}
Result * OnlyTaperConvFunction::getOutput(int outputNo) 
{
    if (outputNo==0) return this->getMyResult("FinalConvFunction");
    else if (outputNo==1) return this->getMyResult("convInfo");
    else if (outputNo==2) return this->getMyResult("ConvSums");
    else throw ImagerException("Only two outputs are available");
    
}
void OnlyTaperConvFunction::resultRead(GAFW::ProxyResult *,int snapshot_no)
{
   // Nothing to do 
}
/*
 void OnlyTaperConvFunction::registerParameters(CppProperties::PropertiesManager & manager)
{
    manager.addPropertyDefenition("taper.operator","The operator name to use for the tapering function",Properties::String);
    manager.addPropertyDefenition("taper.support","Support of tapering function",Properties::Int);
    manager.addPropertyDefenition("taper.sampling","Sampling of taper",Properties::Int);
    manager.addPropertyDefenition("wproj.convFunction.support","TODO",Properties::Int);
    manager.addPropertyDefenition("wproj.convFunction.sampling","TODO",Properties::Int);
    manager.addPropertyDefenition("image.nx","Image length in pixels",Properties::Int);
    manager.addPropertyDefenition("image.ny","Image height in pixels",Properties::Int);
    manager.addPropertyDefenition("image.lintervalsize","The length of pixel in arcsec (or the length of the horizontal interval between each pixel)",Properties::Float);
    manager.addPropertyDefenition("image.mintervalsize","The height of pixel in arcsec (or the length of the horizontal interval between each pixel)",Properties::Float);
    manager.addPropertyDefenition("wproj.wplanes","No of W planes for w-projection",Properties::Int);
}*/
GAFW::Result * OnlyTaperConvFunction::getConvFunction()
{
    this->getOutput(0);
}
GAFW::Result * OnlyTaperConvFunction::getConvFunctionPositionData()
{
    this->getOutput(1);
}
GAFW::Result * OnlyTaperConvFunction::getConvFunctionSumData()
{
    this->getOutput(2);
}
int OnlyTaperConvFunction::getMaxSupport()
{
    //to do
    return 0;
}
int OnlyTaperConvFunction::getSampling()
{
    return this->conf.taper_sampling;
}
float OnlyTaperConvFunction::getWSquareIncrement()
{
    return this->wIncrement;
}
