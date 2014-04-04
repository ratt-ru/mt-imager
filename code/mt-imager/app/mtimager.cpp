/* mtimager.h: main() function and handleHeaders() function. 
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

#include <cstdlib>
#include <fitsio.h>
#include "MTImagerException.h"
#include "GPUafw.h"
#include "WImager.h"
#include "WProjectionConvFunction.h"
#include "OnlyTaperConvFunction.h"
#include "ImageFinalizer.h"
#include "ImagerFactoryHelper.h"

#include "PropertiesManager.h"
#include "Properties.h"
#include "VisibilityManager.h"
#include "ConfigurationManager.h"
#include "ImagerStatistics.h"
#include "ImagerTimeStatistic.h"
#include "DetailedTimeStatistic.h"
#include "GPUOperatorsFactoryHelper.h"
#include <ctime>
#include "mtimager.h"
void initLog(string logfile);
#include <iostream>
using namespace std;
using namespace GAFW;
using namespace mtimager;
using namespace GAFW::Tools::CppProperties;
using namespace mtimager::statistics;


void handleHeaders(Properties params)
{
    if (params.isPropertySet("test")&&params.getBoolProperty("test"))
    {    
        cout << "OK";
        exit(0);
    }
    if (params.isPropertySet("info")&&params.getBoolProperty("info"))
    {    
        cout << "Malta-Imager (mtimager) alpha version" <<endl
             << "Copyright (C) 2013  Daniel Muscat" <<endl
            << "This program is free software: you can redistribute it and/or modify" <<endl
            << "it under the terms of the GNU General Public License as published by" << endl
            << "the Free Software Foundation, either version 3 of the License, or" <<endl
            << "(at your option) any later version."<<endl
            << "This program is distributed in the hope that it will be useful,"<<endl
            << "but WITHOUT ANY WARRANTY; without even the implied warranty of" <<endl
            << "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the" << endl
            << "GNU General Public License for more details." <<endl
            << endl 
            << "You should have received a copy of the GNU General Public License" <<endl
            << "along with this program.  If not, see <http://www.gnu.org/licenses/>."  <<endl
            << endl
            << "Author's contact details can be found at http://www.danielmuscat.com." <<endl; 
        exit(0);
    }
    if (params.isPropertySet("help")&&params.getBoolProperty("help"))
    {    
        cout << "Help will be printed here .. a list of all possible parameters....TODO"<<endl ;
        exit(0);
    }
    
    
    
       
    
    
}
int main(int argc, char** argv) 
{
    Identity mainIdentity("main()","main()");
    ConfigurationManager confMan(argc,argv);
    Properties &params=confMan.getParams();
    //Second phase,, Print headers... Note that below function might cause an exit of the program
    
    handleHeaders(params);
    
    ImageManager * imgMan=NULL ;

    if (!params.isPropertySet("logconf"))
    {
        cout << "logconf parameter is not set.. Can't initililize logger. Exiting" << endl;
        exit(1);
    }
    // Ok Let's try to initilize log4cxx now
    initLog(params.getProperty("logconf"));
   
    
    //Fourth phase...do the job
    ImagerStatistics * imgStat=new ImagerStatistics(confMan.getImagerStatisticsConf());
    //ImagerStatistics imagerStatistics(confMan.getImagerStatisticsConf());
    ImagerStatistics &imagerStatistics=*imgStat;
    confMan.setImagerStatisticsPointer(&imagerStatistics);
    
    Factory * F=new GAFW::GPU::GPUFactory(&imagerStatistics);
    F->registerHelper(new GAFW::GPU::StandardOperators::GPUOperatorsFactoryHelper());
    F->registerHelper(new ImagerFactoryHelper());
    
    VisibilityManager visibilityManager(F,"Visibility Manager",confMan.getVisibilityManagerConf()/*,params*/);
    
    
    
    
    confMan.setVisibilityManager(&visibilityManager);
    
    int noOfChannels=confMan.getNoOfChannelsToPlot();
    //WImager * imager;
    WImager ** wimager=new WImager*[noOfChannels];
    ImageFinalizer ** finalizer=new ImageFinalizer*[noOfChannels];
    ConvolutionFunctionGenerator *gen;
    
    if (confMan.getMode()=="normal")
    {
        gen= new OnlyTaperConvFunction(
                F,
                string("SimpleInterferometricGridding"),
                confMan.getOnlyTaperConvFunctionConf());
    
    }
    else if (confMan.getMode()=="wproj")
    {
        gen=new WProjectionConvFunction(F,
                string("WProjectionConvolutionGenerator"),
                confMan.getWProjectionConvFunctionConf());
        
    }
    else
        throw MTImagerException("Parameter \"mode\" is set to an unsupported value",(void*)NULL,__LINE__,__FILE__);
    WImager::Conf wimager_conf=confMan.getWImagerConf();
    ImageFinalizer::Conf finalizer_conf=confMan.getImageFinalizerConf();
    vector<Result *> imageResVec;
            
    for (int i=0;i<noOfChannels;i++)
    {
        std::stringstream wimager_nickname;
        wimager_nickname<< "Wimager-ChoosenChannel-"<<i; 
        wimager[i]=new WImager(F,wimager_nickname.str(),gen,wimager_conf);
        wimager[i]->setInput(0,visibilityManager.getOutput(VisData::UVW));
        wimager[i]->setInput(1,visibilityManager.getOutput(VisData::VISIBILITY));
        wimager[i]->setInput(2,visibilityManager.getOutput(VisData::FREQUENCY));
        wimager[i]->setInput(3,visibilityManager.getOutput(VisData::WEIGHT));
        wimager[i]->setInput(4,visibilityManager.getOutput(VisData::FLAGS));
        
        std::stringstream finalizer_nickname;
        finalizer_nickname<< "Finalizer-ChoosenChannel-"<<i; 
        finalizer[i]=new ImageFinalizer(F,finalizer_nickname.str(),confMan.getImageFinalizerConf());
        finalizer[i]->setInput(0,wimager[i]->getOutput(0));
        finalizer[i]->setInput(1,wimager[i]->getOutput(1));
        imageResVec.push_back(finalizer[i]->getOutput(0));
        finalizer[i]->getOutput(0)->requireResults();
        
    }
    {
        scoped_detailed_timer t((new DetailedTimerStatistic("ConvFunctionCalulation","N/A",-1)),&imagerStatistics);
        gen->calculateConvFunction();
    }
   
   //ImageFinalizer *fin=new ImageFinalizer(F,"finalizer",confMan.getImageFinalizerConf());
    //fin->getOutput(0)->requireResults();
    while(visibilityManager.nextChannelGroup())
    {   
       
        while(visibilityManager.nextChunk())
        {   
            for (int channelNo=visibilityManager.nextChannel();channelNo!=-1;channelNo=visibilityManager.nextChannel())
            {
                scoped_detailed_timer t((new DetailedTimerStatistic("Gridding Request to GAFW","N/A",channelNo)),&imagerStatistics);
                wimager[channelNo]->calculate();
            }
        }
        vector<int> curChannels=visibilityManager.getCurrentChannels();
        for (vector<int>::iterator channelNo=curChannels.begin();channelNo<curChannels.end();channelNo++)
        {
            wimager[*channelNo]->getOutput(0)->removeReusabilityOnNextUse();  //Avoids useless caching 
            finalizer[*channelNo]->calculate();
            if (imgMan==NULL) //This is a tru as to remove a bit of the load at the begining...loading from hard-disk is expected to be ready here
                imgMan=new ImageManager(confMan.getImageManagerConf(),imageResVec);
            imgMan->nextChannelReady();
            
            
        }
    }
    
    for (int i=0;i<noOfChannels;i++)
    {
        wimager[i]->generateStatistics();
    }
    
    delete imgMan;
    {
    scoped_detailed_timer t((new DetailedTimerStatistic("END","N/A",-1)),&imagerStatistics);
    //imagerStatistics.finalize();
    }
    {
        scoped_detailed_timer t((new DetailedTimerStatistic("DELETION of factory","N/A",-1)),&imagerStatistics);
        delete F;
    }
    
    delete imgStat;
    exit(0);
}

