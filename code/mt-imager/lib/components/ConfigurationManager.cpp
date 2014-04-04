/* ConfigurationManager.cpp:  Implementation of the ConfigurationManager component and class. 
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
#include "ConfigurationManager.h"
#include "PropertiesManager.h"
#include "MTImagerException.h"
#include "WProjectionConvFunction.h"
#include "WImager.h"
#include "ImageFinalizer.h"
#include "measures/Measures/MFrequency.h"
#include "OnlyTaperConvFunction.h"
#include<iostream>
using namespace mtimager;
using namespace GAFW::Tools::CppProperties;
ConfigurationManager::ConfigurationManager(int argc, char** argv) {
    //Temporary
    
    propMan.addPropertyDefenition("conf","Needs to be set to the file default configuration of mtimager",Properties::String);
    propMan.addPropertyDefenition("test","Set this to true (if passing as argument just type -test and the program will print OK to stdout and exit",Properties::Bool);
    propMan.addPropertyDefenition("info","Prints info regarding the mtimager",Properties::Bool);
    propMan.addPropertyDefenition("help","Prints some help nd exists",Properties::Bool);
    propMan.addPropertyDefenition("logconf","Set to the properties file that configures log4cxx, the logging API of the mtimager",Properties::String);
    propMan.addPropertyDefenition("channel","The channel to image. Note that currectly polorizations are regarded as channels",Properties::Int);
    propMan.addPropertyDefenition("channel.frequency","Temporary... Input channel frequency",Properties::Float);
    propMan.addPropertyDefenition("ms","MS File to read",Properties::String);
    propMan.addPropertyDefenition("dataType","",Properties::String);
    propMan.addPropertyDefenition("records","No of records to be submitted to gridder in one calculation",Properties::Int);
    propMan.addPropertyDefenition("mode","The mode to work in: either wproj or normal",Properties::String);
    
    propMan.addPropertyDefenition("taper.operator","The operator name to use for the tapering function",Properties::String);
    propMan.addPropertyDefenition("taper.support","Support of tapering function",Properties::Int);
    propMan.addPropertyDefenition("taper.sampling","Sampling of taper",Properties::Int);
    propMan.addPropertyDefenition("image.nx","Image length in pixels",Properties::Int);
    propMan.addPropertyDefenition("image.ny","Image height in pixels",Properties::Int);
    propMan.addPropertyDefenition("image.lintervalsize","The length of pixel in arcsec (or the length of the horizontal interval between each pixel)",Properties::Float);
    propMan.addPropertyDefenition("image.mintervalsize","The height of pixel in arcsec (or the length of the horizontal interval between each pixel)",Properties::Float);
    
    propMan.addPropertyDefenition("wproj.convFunction.support","TODO",Properties::Int);
    propMan.addPropertyDefenition("wproj.convFunction.sampling","TODO",Properties::Int);
    propMan.addPropertyDefenition("wproj.wplanes","No of W planes for w-projection",Properties::Int);
    propMan.addPropertyDefenition("outputFITS", "The FITS file to which output image shall be saved.", Properties::String);
    propMan.addPropertyDefenition("parallel_channels","The maximum number of channels calculated in parallel",Properties::Int);
    propMan.addPropertyDefenition("field","The field to choose for imaging. The parameter is considered only when the MS file contains more then 1 field",Properties::Int);
    propMan.addPropertyDefenition("statistics_file.engine","CSV file to save Engine statistics",Properties::String);
    propMan.addPropertyDefenition("statistics_file.gridding","CSV file to save gridding statistics",Properties::String);
    propMan.addPropertyDefenition("statistics_file.main","CSV file to save main statistics (timelines of various actions)",Properties::String);
    propMan.addPropertyDefenition("psf","When set to true generates PSFs by setting all visibility values to 1",Properties::Bool);
    propMan.loadPropertyArgs(params,argc, argv);


#define DEF(x) #x
#define DEF2(x) DEF(x)
    
    //DEFAULT_CONF_FILE is set through cmake
    if (!params.isPropertySet("conf"))
    {

        params.setProperty("conf",string(DEF2(DEFAULT_CONF_FILE)));
    }
    
#undef DEF
#undef DEF2
    if (params.isPropertySet("conf"))
    {
        propMan.loadPropertyFile(params, params.getProperty("conf"));
        propMan.loadPropertyArgs(params,argc, argv);
    }
    this->l_increment=casa::Quantity(this->params.getFloatProperty("image.lintervalsize"),"arcsec");
    this->m_increment=casa::Quantity(this->params.getFloatProperty("image.mintervalsize"),"arcsec");
    this->l_pixels=this->params.getIntProperty("image.nx");
    this->m_pixels=this->params.getIntProperty("image.ny");
    this->channels.push_back(this->params.getIntProperty("channel"));
    this->outputFITSFileName=this->params.getStringProperty("outputFITS");
    
    this->taper.operator_name=this->params.getStringProperty("taper.operator");
    this->taper.support=this->params.getIntProperty("taper.support");
    
    this->mode=this->params.getStringProperty("mode");
    if (this->mode=="normal")
    {
        //Nothing to do 
        this->taper.sampling=this->params.getIntProperty("taper.sampling");
    }
    else if (this->mode=="wproj")
    {
        //Ok we need two other parmeters
        //this->wprojection.functionMinimumSupport=1;//this->params.getIntProperty("wproj.convFunction.support");
        this->wprojection.sampling=this->params.getIntProperty("wproj.convFunction.sampling");
        this->wprojection.wplanes=this->params.getIntProperty("wproj.wplanes");
    }
    else
        throw ImagerException(string("Mode set to: ")+this->mode+ string ( " which is unkown"));
    this->visManager=NULL;
    

}

ConfigurationManager::~ConfigurationManager() {
}
void ConfigurationManager::setVisibilityManager(VisibilityManager * visMan)
{
    this->visManager=visMan;
}

ImageManager::Conf ConfigurationManager::getImageManagerConf()

{
    if (this->visManager==NULL)
        throw ImagerException("Visibility Manager not yet set.. Unable to set Image Manager Configuration");
    ImageManager::Conf conf;
    conf.l_increment=this->l_increment.get("rad").getValue();
    conf.m_increment=this->m_increment.get("rad").getValue();
    conf.lpixels=this->l_pixels;
    conf.mpixels=this->m_pixels;


    conf.freqs=this->visManager->getChoosenChannelFrequencies(); //We work only on the first spw
    conf.obsEpoch;  //TODO
    conf.obsFreqRef=(casa::MFrequency::Types)conf.freqs[0].getRef().getType(); 
    conf.observer="Unimplimented";
    conf.phaseCenter=this->visManager->getPhaseCentre(); 
    conf.restFreq=casa::Quantity(0,"Hz");
    conf.stat=this->statisticsManager;
    conf.stokes.push_back(casa::Stokes::I);  
    
    switch (this->visManager->getNoOfPolarizations())
    {
        case 1:
            break;
        case 2:
            if (this->visManager->getPolarizationType()==PolarizationType::Linear)
                conf.stokes.push_back(casa::Stokes::Q);
            else
                conf.stokes.push_back(casa::Stokes::V);
            break;
        case 4:
            conf.stokes.push_back(casa::Stokes::Q);
            conf.stokes.push_back(casa::Stokes::U);
            conf.stokes.push_back(casa::Stokes::V);
            break;
        default:
            throw ImagerException("No Of polarizations is not supported");
    }
    
    conf.telescope=this->visManager->getTelescopeName();
    conf.outputFITSFileName=this->outputFITSFileName;         



    conf.initialized=true;
    return conf;
}

WImager::Conf ConfigurationManager::getWImagerConf()
{
    if (this->visManager==NULL)
        throw ImagerException("Visibility Manager not yet set.. Unable to set WImager Configuration");
    WImager::Conf conf;
    conf.no_of_polarizations=this->visManager->getNoOfPolarizations();
    conf.u_pixels=this->l_pixels;
    conf.v_pixels=this->m_pixels;
    conf.u_increment=1.0/(this->l_increment.get("rad").getValue()*this->l_pixels);
    conf.v_increment=1.0/(this->m_increment.get("rad").getValue()*this->m_pixels);
    conf.statisticsManager=this->statisticsManager;
    return conf;
}
WProjectionConvFunction::Conf ConfigurationManager::getWProjectionConvFunctionConf()
{
    if (this->visManager==NULL)
        throw ImagerException("Visibility Manager not yet set.. Unable to set WImager Configuration");
    WProjectionConvFunction::Conf conf;
    if (mode!="wproj") throw ImagerException("Configuration Function called when mode is not wproj");
    conf.conv_function_sampling=this->wprojection.sampling;
    //conf.conv_function_support=this->wprojection.functionMinimumSupport;
    conf.image_total_l=this->l_increment.get("rad").getValue()*(double)this->l_pixels;
    conf.image_total_m=this->m_increment.get("rad").getValue()*(double)this->m_pixels;
    conf.image_l_increment=this->l_increment.get("rad").getValue();
    conf.taper_operator=this->taper.operator_name;
    conf.taper_support=this->taper.support;
    conf.wplanes=this->wprojection.wplanes;
    conf.standard_method=true;
    if (this->l_pixels<this->m_pixels)
        conf.img_min_dim=this->l_pixels;
    else
        conf.img_min_dim=this->m_pixels;
    
    return conf;
    
}
OnlyTaperConvFunction::Conf ConfigurationManager::getOnlyTaperConvFunctionConf()
{
    OnlyTaperConvFunction::Conf conf;
    conf.taper_operator=this->taper.operator_name;
    conf.taper_sampling=this->taper.sampling;
    conf.taper_support=this->taper.support;
    return conf;
}
ImageFinalizer::Conf ConfigurationManager::getImageFinalizerConf()
{
    ImageFinalizer::Conf conf;
    conf.image_nx=this->l_pixels;
    conf.image_ny=this->m_pixels;
    conf.taper_operator=this->taper.operator_name;
    conf.polType=this->visManager->getPolarizationType();
    
    conf.taper_support=this->taper.support;
    if (mode=="wproj")
        conf.conv_function_sampling=this->wprojection.sampling;
    else
        conf.conv_function_sampling=this->taper.sampling;
    return conf;
}
string ConfigurationManager::getMode()
{
    return this->mode;
}
mtimager::statistics::ImagerStatistics::Conf ConfigurationManager::getImagerStatisticsConf()
{
    mtimager::statistics::ImagerStatistics::Conf conf;
    conf.mainStatisticsFile=this->params.getProperty("statistics_file.main");
    conf.griddingStatisticsFile=this->params.getProperty("statistics_file.gridding");
    conf.engineStatisticsFile=this->params.getProperty("statistics_file.engine");
    conf.nx=this->l_pixels;
    conf.inc=this->l_increment.getValue();
    conf.sampling=this->taper.sampling;
    conf.support=this->taper.support;
    
    return conf;
            
}
void ConfigurationManager::setImagerStatisticsPointer(mtimager::statistics::ImagerStatistics *statisticsManager)
{
    this->statisticsManager=statisticsManager;
}
VisibilityManager::Conf ConfigurationManager::getVisibilityManagerConf()
{
    VisibilityManager::Conf conf;
    conf.statisticsSystem=this->statisticsManager;
    conf.dataType=params.getStringProperty("dataType");
    conf.maxRecords=params.getIntProperty("records");
    if (params.getIntProperty("channel")==-1)
        conf.allChannels=true;
    else
    {
        conf.channels.push_back(params.getIntProperty("channel"));
        conf.allChannels=false;
    }
    
    conf.dataFileNames.push_back(params.getStringProperty("ms"));
    conf.parallel_channels=params.getIntProperty("parallel_channels");
    conf.field=params.getIntProperty("field");
    conf.doPSF=params.getBoolProperty("psf");
    return conf;
}
int ConfigurationManager::getNoOfChannelsToPlot()
{
    if (this->visManager==NULL) throw ImagerException("Visibility manager not yet initialised");
    return this->visManager->getChoosenChannelFrequencies().size();
}
