/* WProjectionConvFunction.cpp:  Implementation of the WProjectionConvFunction GAFW Module. 
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

#include "WProjectionConvFunction.h"
#include "WPlaneConvFunctionCalculator.h"
//#include "WImager_backup.h"
//#include "ImageDefinition.h"
#include "MTImagerException.h"
#include <sstream>
#include <iostream>
#include <iomanip>
using namespace GAFW;
using namespace std;
using namespace mtimager;
using namespace GAFW::Tools::CppProperties;

void print2D(GAFW::Result *r)
{
    //return;
    //int y=0;
    int b=r->getLastSnapshotId();
    std::vector<unsigned int> pos;
    pos.push_back(0);
    pos.push_back(0);
    pos.push_back(0);
    
    for (int y=0;y<r->getDimensions(b).getNoOfRows();y++) 
    {    for (int x=0; x<r->getDimensions(b).getNoOfColumns();x++)
        {    ValueWrapper<complex<float> > c;
                pos[0]=y;
                pos[1]=x;
                r->getValue(b,pos,c);
        
              cout << c.value <<'\t';
        }
        cout << "END " << y << '\n';
        
    }          
    
    pos[0]=r->getDimensions(b).getNoOfRows()/2;
    pos[1]=r->getDimensions(b).getNoOfColumns()/2;
    ValueWrapper<complex<float> > c;
    r->getValue(b,pos,c);
    cout <<"CENTRE: " << c.value <<endl;
    
    
    cout.flush();
     
}
void print1D(GAFW::Result *r)
{
    //return;
    //int y=0;
    std::vector<unsigned int> pos;
    pos.push_back(0);
    pos.push_back(0);
    pos.push_back(0);
    int b=r->getLastSnapshotId();
    
    for (int x=0; x<r->getDimensions(b).getNoOfColumns();x++)
        {    ValueWrapper<complex<float> > c;
                
                pos[0]=x;
                r->getValue(b,pos,c);
        if (x%49==0) cout << endl;
              cout << c.value.real() <<'\t';
              
        }
             
    
     
}
//WProjectionConvFunction::WProjectionConvFunction(GAFW::Factory *factory,std::string nickname,  CppProperties::Properties &oparams, ImageManagerOld & imageMan):Module(factory,nickname,"W-Projection Imager"),Identity(nickname,"W-Projection Imager"),imagedef(imageMan)
WProjectionConvFunction::WProjectionConvFunction(GAFW::Factory *factory,std::string nickname, WProjectionConvFunction::Conf conf):Module(factory,nickname,"W-Projection Imager"),conf(conf)
{
    logDebug(other,"Initialising...");
    logDebug(other,"Initialising the the convolution function calculator module ");
    this->convCalculator=new WPlaneConvFunctionCalculator(factory,nickname+".calculator",conf);
   // paramLoader(oparams);
    //if(!this->imagedef.isOk()) throw ImagerException("An image parameter is not set well. Please check image.nx, image.ny, image.lintervalsize & image.mintervalsize");
    
    //A bit of logging for info  //Have to change the way as it is too cumbersome
    
    
    stringstream debug;
    debug << "W Projection will using " << this->conf.wplanes << " planes"; 
    logInfo(other,debug.str());
    
    float maxW=0.25/this->conf.image_l_increment;
    //float maxW=2000; //OK GODDD... but we need to put accuracy
    debug.str("");
    //std::cout << "Estimated maximum possible W set to  " << maxW << " wavelengths."<<endl; 
    logInfo(other,debug.str());
    
    if (this->conf.wplanes==1) this->wIncrement=1.0; //Value not important in this case but don't let it be 0
    else  this->wIncrement=maxW/(float)((this->conf.wplanes-1)*(this->conf.wplanes-1));
    debug.str("");
    //std::cout << "Increment per plane is " << wIncrement << " wavelengths per pixel"<<endl;
    logInfo(other,debug.str());
    
    this->requestMyArray("BoundToNormalizedUnorderedConvFunction")->bind_to(this->convCalculator->getConvFunction());
    this->requestMyOperator("convFuncOrganise","ConvFunctionReOrganize","BoundToNormalizedUnorderedConvFunction");
    ParameterWrapper<int> int_parameter;
    this->getMyOperator("convFuncOrganise")->setParameter(int_parameter.setNameAndValue("ConvolutionFunction.sampling",this->conf.conv_function_sampling));
    




    
    //Last but not least we need to have a structure that generates the final one big streamed "convolution" function
    
    
    this->requestMyArray("convInfo",ArrayDimensions(2,this->conf.wplanes,2),DataType<int>());
    //this->getMyArray("convInfo")->createStore(real_int);
    
    this->requestMyOperator("convFunctionAmalgamator","ArraysToVector")->a("AmalgamatedConvFunction");
    
    
    //Create all input Arrays for above...withequals and everything
    this->getMyArray("AmalgamatedConvFunction")->getResults()->reusable();
    GAFW::ArrayOperator *amalgamator=this->getMyOperator("convFunctionAmalgamator");
    this->requestMyOperator("SumConvolutionFunctions","SumConvolutionFunctions","convInfo","AmalgamatedConvFunction")->a("ConvSums");
    this->getMyOperator("SumConvolutionFunctions")->setParameter(int_parameter.setNameAndValue("sampling",this->conf.conv_function_sampling));
   
    this->getMyArray("ConvSums")->getResults()->reusable();
    
     
    this->convFunctions.clear(); ///just in case
    for (int i=0;i<this->conf.wplanes;i++)
    {
       stringstream name;
       name<< "ConvolutionFunction"<< i;
       GAFW::Array *convArray=this->requestMyArray(name.str());
       convArray->getResults()->reusable();
       this->convFunctions.push_back(convArray);
       GAFW::Array *boundArray=this->requestMyArray(string("Bound")+name.str());
       boundArray->bind_to(convArray->getResults());
       amalgamator->setInput(boundArray);
       
    }
    logDebug(other,"Initialisation complete");
    
}
WProjectionConvFunction::~WProjectionConvFunction()
{
    
}/*
void WProjectionConvFunction::paramLoader(CppProperties::Properties& oparams)
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
void WProjectionConvFunction::reset()
{
    
}
void WProjectionConvFunction::calculateConvFunction()
{
    this->calculate();
}

void WProjectionConvFunction::calculate()
{
    ParameterWrapper<int> int_parameter;
    stringstream debug;
    
    //Ok let's begin......
    //We first need to make a "special" calculation as to calulate the "divisor"
    int currentSupport;
    if(conf.standard_method)
    {
        currentSupport=this->conf.img_min_dim/this->conf.conv_function_sampling;
        if (currentSupport%2==0) currentSupport++;
        
        
    }
    else
    {
//        currentSupport=this->conf.conv_function_support;
        
    }
    
    this->convCalculator->setTrialSupport(currentSupport);
    complex<double> divisor=this->convCalculator->getW0OriginValue();
    debug << "Divisor set to " <<divisor;
    logInfo(other,debug.str());
    debug.str("");
    
    if (divisor.imag()>1e-3) 
    {
        stringstream ss;
        ss<< "The origin point for convolution function at w=0 has a non-zero imaginary value. Value of origin is " << divisor; 
        throw ImagerException(ss.str());
    }
    float multiplier=1/divisor.real(); 
    //multiplier=1;
    debug << " multiplier is set to  " <<  multiplier*(currentSupport)*(currentSupport);
    logWarn(other,debug.str());
    
    
    //res->doNotRequireResults();//No need anymore to have such data
    vector<unsigned int> supportVec;
    vector<unsigned int> zeropos;
    zeropos.push_back(0);
            
    GAFW::Array *convInfo=this->getMyArray("convInfo");
    GAFW::ArrayOperator *organiseOperator=this->getMyOperator("convFuncOrganise");
    ValueWrapper<int> planeConvIndex=0;
    ValueWrapper<int> support=0;
//    ParameterWrapper<int> int_parameter;
    for (int wplane=0; wplane<this->conf.wplanes;wplane++)
    {
        //this->getMyArray("NormalizedUnorderedConvFunction")->getResults()->requireResults();
        float w=wplane*wplane*wIncrement;
        this->convCalculator->setW(w);
        this->convCalculator->setNormalizer(multiplier);
        for (;;) //It will stop when a break occurs
        {
            this->convCalculator->setTrialSupport(currentSupport);
            this->convCalculator->calculate();
            support.value=this->convCalculator->getCalculatedSupport();
            //support_res->getValue(id,zeropos,support);
            
            if (support.value>(currentSupport-2))
                support.value=currentSupport-2;
            //if (support<currentSupport-2)
            //{
                //cout <<"support for w=" << w << " is "<<support.value<<endl;
                organiseOperator->setOutput(0,this->convFunctions[wplane]);
                organiseOperator->setParameter(int_parameter.setNameAndValue("ConvolutionFunction.support",support.value));
                this->convFunctions[wplane]->getResults()->requireResults();
                int id=this->convFunctions[wplane]->getResults()->calculate();
                supportVec.push_back(support.value);
                vector<unsigned int> p;
                p.push_back(wplane);
                p.push_back(0);
                convInfo->setValue(p,support);
                p[1]=1;
                convInfo->setValue(p,planeConvIndex);
                this->convFunctions[wplane]->getResults()->waitUntilDataValid();
                planeConvIndex.value+=this->convFunctions[wplane]->getResults()->getDimensions(id).getX(); 
                //support_res->calculate();
                break;
            //}
            //else
            //{
            //    if (this->conf.standard_method)
            //    {
            //        throw ImagerException("Support is too large");
            //        break;
            //    }
            //    currentSupport+=2;
            //}
        }
    }
    //OK ready..Now we amalgamate all results
    //this->getMyArray("AmalgamatedConvFunction")->getResults()->requireResults();
    stringstream iiii;
    iiii<< "Supports are ";
    for (vector<unsigned int>::iterator i=supportVec.begin();i<supportVec.end();i++)
        iiii<< *i<< " ";
    this->logInfo(other,iiii.str());
   
    this->getMyArray("ConvSums")->getResults()->calculate();
    

}
void WProjectionConvFunction::setInput(int inputNo, GAFW::Result *res)
{
    throw ImagerException("No inputs expected");
    
}
Result * WProjectionConvFunction::getOutput(int outputNo) 
{
    if (outputNo==0) return this->getMyArray("AmalgamatedConvFunction")->getResults();
    else if (outputNo==1) return this->getMyArray("convInfo")->getResults();
    else if (outputNo==2) return this->getMyArray("ConvSums")->getResults();
    else throw ImagerException("Only two outputs are available");
    
}
void WProjectionConvFunction::resultRead(GAFW::ProxyResult *,int snapshot_no)
{
   // Nothing to do 
}
/*
 void WProjectionConvFunction::registerParameters(CppProperties::PropertiesManager & manager)
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
GAFW::Result * WProjectionConvFunction::getConvFunction()
{
    return this->getOutput(0);
}
GAFW::Result * WProjectionConvFunction::getConvFunctionPositionData()
{
    return this->getOutput(1);
}
GAFW::Result * WProjectionConvFunction::getConvFunctionSumData()
{
    return this->getOutput(2);
}
int WProjectionConvFunction::getMaxSupport()
{
    //to do
    return 0;
}
int WProjectionConvFunction::getSampling()
{
    return this->conf.conv_function_sampling;
}
float WProjectionConvFunction::getWSquareIncrement()
{
    return this->wIncrement;
}
       