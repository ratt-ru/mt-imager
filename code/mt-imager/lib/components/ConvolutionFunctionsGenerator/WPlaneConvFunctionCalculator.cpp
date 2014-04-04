/* WPlaneConvFunctionCalculator.cpp:  Implementation of the WPlaneConvFunctionCalculator GAFW Module. 
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

#include "WPlaneConvFunctionCalculator.h"
#include "WProjectionConvFunction.h"
#include "MTImagerException.h"
#include <sstream>
#include <iostream>
#include <iomanip>
using namespace GAFW;
using namespace std;
using namespace mtimager;
using namespace GAFW::Tools::CppProperties;

WPlaneConvFunctionCalculator::WPlaneConvFunctionCalculator(GAFW::Factory *factory,std::string nickname, struct WProjectionConvFunction::Conf conf):Module(factory,nickname,"W-Plane ConvFunction Calculator"),conf(conf)
{
    logDebug(other,"Initialising...");
    
    
    ParameterWrapper<int> int_parameter;
    ParameterWrapper<bool> bool_parameter;
    ParameterWrapper<double> double_parameter;
    ParameterWrapper<float> float_parameter;
    this->requestMyOperator("taper",this->conf.taper_operator)->a("taper_lm_correction");
    this->getMyOperator("taper")->setParameter(int_parameter.setNameAndValue("support",this->conf.taper_support));
    this->getMyOperator("taper")->setParameter(int_parameter.setNameAndValue("sampling",1));// TODO this operator signored sampling but still validates against it
    this->getMyOperator("taper")->setParameter(bool_parameter.setNameAndValue("lmPlanePostCorrection",true));
    this->getMyOperator("taper")->setParameter(bool_parameter.setNameAndValue("generate",true));
    this->getMyOperator("taper")->setParameter(bool_parameter.setNameAndValue("double-precision",false));
    this->requestMyOperator("preFFTConvFunction","WProjPreFFTConvFunc","taper_lm_correction")->a("preFFTConvFunc");
    
    this->getMyOperator("preFFTConvFunction")->setParameter(int_parameter.setNameAndValue("sampling",this->conf.conv_function_sampling));
    this->getMyOperator("preFFTConvFunction")->setParameter(double_parameter.setNameAndValue("total_l",this->conf.image_total_l));
    this->getMyOperator("preFFTConvFunction")->setParameter(double_parameter.setNameAndValue("total_m",this->conf.image_total_m));
    this->getMyOperator("preFFTConvFunction")->setParameter(bool_parameter.setNameAndValue("lwimager-removeedge",false));
    
    this->requestMyOperator("convFuncInverseFFT","multiFFT2D","preFFTConvFunc")->a("preShiftedConvFunc");
    
    this->requestMyOperator("convFuncZeroShift","ZeroToCentre2DShift","preShiftedConvFunc")->a("ConvolutionFunctionNotOrdered");
    
    this->requestMyOperator("convNormalizer","SingleMultiply","ConvolutionFunctionNotOrdered")->a("NormalizedUnorderedConvFunction");
    
    this->requestMyOperator("supportfind","ConvFunctionSupportFind","NormalizedUnorderedConvFunction")->a("Support");
    this->getMyOperator("supportfind")->setParameter(int_parameter.setNameAndValue("ConvolutionFunction.sampling",this->conf.conv_function_sampling));
    this->getMyOperator("supportfind")->setParameter(float_parameter.setNameAndValue("ConvolutionFunction.takeaszero",1e-3f));
    this->getMyArray("Support")->getResults()->requireResults();
    this->getMyArray("NormalizedUnorderedConvFunction")->getResults()->reusable();
    
    logDebug(other,"Initialisation complete");
    
}
WPlaneConvFunctionCalculator::~WPlaneConvFunctionCalculator()
{
    
}
void WPlaneConvFunctionCalculator::reset()
{
    
}
void WPlaneConvFunctionCalculator::calculate()
{
    stringstream debug;
    this->getMyResult("Support")->calculate();

}
void WPlaneConvFunctionCalculator::setInput(int inputNo, GAFW::Result *res)
{
    throw ImagerException("No inputs expected");
    
}
Result * WPlaneConvFunctionCalculator::getOutput(int outputNo) 
{
    if (outputNo==1) return this->getMyResult("Support");
    if (outputNo==0) return this->getMyResult("NormalizedUnorderedConvFunction");
    else throw ImagerException("Only two outputs are available");
    
}
void WPlaneConvFunctionCalculator::setW(double w)
{
    ParameterWrapper<float> float_parameter("w",(float)w);
    this->getMyOperator("preFFTConvFunction")->setParameter(float_parameter);
}

void WPlaneConvFunctionCalculator::setTrialSupport(int support)
{
    ParameterWrapper<int> int_parameter;
    this->getMyOperator("taper")->setParameter(int_parameter.setNameAndValue("nx",support));
    this->getMyOperator("taper")->setParameter(int_parameter.setNameAndValue("ny",support));
    this->getMyOperator("preFFTConvFunction")->setParameter(int_parameter.setNameAndValue("support",support));
   
}
  
void WPlaneConvFunctionCalculator::resultRead(GAFW::ProxyResult *,int snapshot_no)
{
   // Nothing to do 
}
GAFW::Result * WPlaneConvFunctionCalculator::getConvFunction()
{
    this->getOutput(0);
}
int WPlaneConvFunctionCalculator::getCalculatedSupport()
{
    ValueWrapper<int> support;
    vector <unsigned int> zeropos;
    zeropos.push_back(0);
    this->getMyResult("Support")->waitUntilDataValid();
    
    this->getMyResult("Support")->getValue(this->getMyResult("Support")->getLastSnapshotId(),zeropos,support);
    return support.value;
}

complex<double> WPlaneConvFunctionCalculator::getW0OriginValue()
{
    this->setW(0.0);
    Result *res=this->getMyResult("preShiftedConvFunc");
    res->requireResults();
    res->calculate();
    res->doNotRequireResults();
    res->waitUntilDataValid();
    vector<unsigned int> zeropos;
    zeropos.push_back(0);
    zeropos.push_back(0);
    ValueWrapper<complex<float> > W0Value;
    res->getValue(res->getLastSnapshotId(),zeropos,W0Value);
    return W0Value.value;
}

void WPlaneConvFunctionCalculator::setNormalizer(float normalizer)
{
    ParameterWrapper<float> normalizer_parameter("multiplier.value",normalizer);
    this->getMyOperator("convNormalizer")->setParameter(normalizer_parameter);
}