/* VisibilityManager.cpp: Implementation of the VisibilityManager component and class. 
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

#include "mtimager.h"
#include <sstream>
#include <iomanip>
#include <cstring>
#include <bits/stl_vector.h>

using namespace mtimager;
using namespace std; 
using namespace GAFW;
using namespace GAFW::Tools::CppProperties;

VisibilityManager::VisibilityManager(GAFW::Factory *factory,std::string nickname, Conf conf)
        :Identity(nickname,"Visibility Manager"),FactoryAccess(factory)
{
    LogFacility::init();
    FactoryAccess::init();
       
    this->getFactory()->registerIdentity(this);
    this->logDebug(other,"Initialising the visibility manager");
    this->conf=conf;
    if (this->conf.dataType=="ms")
    {
        factory->logDebug(other,"One Single MS file to be loaded");
        this->dataSets.push_back(new MSVisibilityDataSetAsync(this->getFactory(),this->conf.dataFileNames[0],this->conf.statisticsSystem,this->conf.maxRecords,this->conf.parallel_channels,this->conf.field,this->conf.doPSF));
    }
    else
        throw ImagerException(string("Unknown data type format ")+this->conf.dataType);
    
    //For the time being we hard code some confs
        
    this->outputValid=false;
    
    this->noOfPolarizations=dataSets[0]->getNoOfPolarizations();
    stringstream s;
    s<< this->noOfPolarizations << " polarizations found.";
    this->logDebug(other,s.str());
    this->logDebug(other,"Loading channel Frequencies");
    this->measChannelFreqs=this->dataSets[0]->getChannelFrequencies();
    int noOfChannels=this->measChannelFreqs.size();
    s << "Found "<< noOfChannels << " channels";
    this->logDebug(other,s.str());
    s.str(string());
    if (this->measChannelFreqs.size()==0) throw ImagerException("No Channels Found!");
    
    
    this->polType=this->dataSets[0]->getPolarizationType();
   //Now that we know all channels in the file we need to check with conf 
    //that the user requested known channels and then inform teh MS reader that we will need only these channels
    vector<bool> chosenChannels;
    //First we check
    //if (this->conf.allChannels==false) 
    //{
    //NOTE:: conf.channels is assumed to be ordered and epmyty if all channels is true This is a task for the configuration manager
     vector<int>::iterator i=conf.channels.begin();
    // this->channels.push_back(-1); //first entry -1
     for (int channelNo=0;channelNo<(int)this->measChannelFreqs.size();channelNo++)
     {
         bool channelRequired=this->conf.allChannels;
         if (i!=this->conf.channels.end())
         {
             if (channelNo==*i)
             {
                 //this channel is required..
                 channelRequired=true;
                 i++;
             }

         }
         chosenChannels.push_back(channelRequired);
         if (channelRequired) 
         {
             this->channels.push_back(channelNo);
             this->measChosenChannelFreqs.push_back(this->measChannelFreqs[channelNo]);
         }

     }
     if (i!=conf.channels.end())
     {
         s.str("");
         s << "Channel No " << *i <<" does not exist";
         throw ImagerException(s.str());
     }

        
        
    
    /* 
    if (this->conf.channel>=noOfChannels) 
    {
        s<< "Channel no " << this->conf.channel << " does not exist";
        throw ImagerException(s.str());
    }
    s.str();
    s << "Chosen channel is "<< this->conf.channel;// << " with " << this->measChannelFreqs[this->channel];
    this->logDebug(other, s.str());
     
    vector<bool> channelChoose;
    for (int channelNo=0;channelNo<noOfChannels;channelNo++)
    {
        if (this->conf.channel==channelNo)
            channelChoose.push_back(true);
        else
            channelChoose.push_back(false);
    }
     */
    this->dataSets[0]->setChannelChoice(chosenChannels);
    //The above prepares the data Set
    
    //We need to know the position of telescope.. we fisrt ask for the 
    //name and query casacore for the position
    this->telescopeName=dataSets[0]->getTelescopeName();
    this->logDebug(other,string("Telescope Name is ")+this->telescopeName);
    if (!casa::MeasTable::Observatory(this->telescopePosition,casa::String(this->telescopeName)))
        throw ImagerException("Query to casa::MeasTable::Observatory() for telescope position was unsuccessful");
    s.str(string());
    s<< "Telescope Position: " << this->telescopePosition;
    this->logDebug(other,s.str());
    
    //For now phase centre is set to that of forst dataset 
    this->phaseCentre=this->dataSets[0]->getPhaseCentre();
    
    this->logDebug(other,"Initialising internal outputs");
    
    for (int i=0 ; i<VisData::TotalOutputs;i++)
    {
        //this->outputArrays[i]=this->requestMyArray(VisData::names[i]);
        this->outputResult[i]=this->requestMyProxyResult(VisData::names[i]);
    }
    
    this->dataSets[0]->initAsyncLoading();
    this->currentDataSet=0;
    this->currentChunkNo=-1;
    //this->currentChannel=this->channels.begin();
    this->channelNoForNextGroup=0;
    this->logDebug(other,"Initialisation successfully completed");
    
    
  
}
VisibilityManager::~VisibilityManager()
{
  //  this->logInfo(other,"deleting");
    //TODO
}
enum PolarizationType::Type VisibilityManager::getPolarizationType()
{
    return this->polType;
}
bool VisibilityManager::nextChunk()
{   
    
    this->currentChunkNo++;
    this->currentChannel=this->currentChannelGroup.begin();
    if (this->dataSets[0]->getNoOfChunks()<=this->currentChunkNo)
    {
        return false;
    }
    for (int i=0;i<(int)VisData::TotalOutputs;i++)
    {
        //this->dataSets[0]->loadArray((VisData::DataType)i,this->currentChunkNo,-1,this->outputArrays[i]);
        GAFW::Array *array=this->dataSets[0]->getArray((VisData::DataType)i,this->currentChunkNo,-1);
        if (array!=NULL)
        {
            this->outputResult[i]->setBind(array->getResults());
        }
                
    }
    return true;
}
int VisibilityManager::nextChannel()
{
    if (this->currentChannel!=this->currentChannelGroup.end())
        this->currentChannel++;
    if (this->currentChannel==this->currentChannelGroup.end())
        return -1; 
    for (int i=0;i<(int)VisData::TotalOutputs;i++)
    {
        //this->dataSets[0]->loadArray((VisData::DataType)i,this->currentChunkNo,this->currentChannel->second,this->outputArrays[i]);
        GAFW::Array *array=this->dataSets[0]->getArray((VisData::DataType)i,this->currentChunkNo,this->currentChannel->second);
        if (array!=NULL)
        {
            this->outputResult[i]->setBind(array->getResults());
        }
        
    }
    return this->currentChannel->first;
}
bool VisibilityManager::nextChannelGroup()
{
   
    this->currentChunkNo=-1;
    this->currentChannelGroup.clear();
    if (this->channelNoForNextGroup==this->channels.size()) return false;
    
    this->currentChannelGroup.push_back(pair<int,int>(-1,-1));
    for (int i=0;i<this->conf.parallel_channels;i++)
    {
        this->currentChannelGroup.push_back(pair<int,int>(this->channelNoForNextGroup,this->channels[this->channelNoForNextGroup]));
        this->channelNoForNextGroup++;
        if (this->channelNoForNextGroup==this->channels.size()) break;
    
    }
    this->currentChannel=this->currentChannelGroup.begin();
    return true;

}
std::vector<int> VisibilityManager::getCurrentChannels()
{
    vector<int> ret;
    for (vector<pair<int,int> >::iterator i=this->currentChannelGroup.begin()+1; i<this->currentChannelGroup.end();i++)
    {
        ret.push_back(i->first);
    }    
    return ret;
}



GAFW::Result * VisibilityManager::getOutput(enum mtimager::VisData::DataType outputNo)
{
    //return this->outputArrays[outputNo]->getResults();
    return this->outputResult[outputNo];
}

std::vector<casa::MFrequency> VisibilityManager::getChannelFrequencies()
{
    return this->measChannelFreqs;
}
std::vector<casa::MFrequency> VisibilityManager::getChoosenChannelFrequencies()
{
    return this->measChosenChannelFreqs;
}

int VisibilityManager::getNoOfPolarizations()
{
    return this->noOfPolarizations;
}
casa::MDirection VisibilityManager::getPhaseCentre()
{
    return this->phaseCentre;
}
std::string VisibilityManager::getTelescopeName()
{
    return this->telescopeName;
}

