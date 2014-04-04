/* MSVisibilityDataSetAsync.cpp: Implementation of the MSVisibilityDataSetAsync  class. 
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

#include "statistics/DetailedTimeStatistic.h"
#include "MSVisibilityDataSetAsync.h"
#include <tables/Tables/Table.h>
#include <tables/Tables/ExprNode.h>


using namespace mtimager;
using namespace casa;
using namespace GAFW;

 std::string MSVisibilityDataSetAsync::convertDots(std::string str)
{
    std::stringstream ss;
    for (std::string::iterator i=str.begin();i<str.end();i++)
    {
        char f=*i;
        if (f=='.') f='_';
        ss<<f;
        
    }
    return std::string(ss.str());
}

MSVisibilityDataSetAsync::MSVisibilityDataSetAsync(GAFW::Factory * factory,std::string msFileName,GAFW::FactoryStatisticsOutput *statisticManager,int chunkLength,int parallel_channels,int field,bool doPSF)
:Identity(MSVisibilityDataSetAsync::convertDots(string("MS Name:")+msFileName),string("MSDataVisibilityAsync")),FactoryAccess(factory)
{
    LogFacility::init();
    FactoryAccess::init();
       
    this->getFactory()->registerIdentity(this);
    scoped_detailed_timer t((new DetailedTimerStatistic(this,"INIT ASYNC","N/A",-1)),statisticManager);
    this->doPSF=doPSF;   
    this->parallel_channels=parallel_channels;
    this->statisticManager=statisticManager;
    this->chunkLength=chunkLength;
    this->noOfChunks=-1;
    this->lastChunkLength=-1;
    this->ms=new MeasurementSet(msFileName);
    
    ROMSFieldColumns field_table(this->ms->field());
       
    if (field_table.nrow()==1)
    {
        this->phaseCentre=field_table.phaseDirMeas(0);
        this->mainCols=new casa::ROMSMainColumns(*this->ms); 
    }
    else if (field_table.nrow()>1)
    {   
        this->phaseCentre=field_table.phaseDirMeas(field);
        tab=new Table(msFileName);
        tabselect=new Table(tab->operator()(tab->col("FIELD_ID")==field));
        this->mainCols=new casa::ROMSMainColumns(*tabselect);

    }
    MEpoch meas;
    this->epochType=(casa::MEpoch::Types)this->mainCols->timeMeas().getMeasRef().getType();
    this->noOfPolarizationsTypes=this->ms->polarization().nrow(); //Most probably not right
    this->noOfPolarizations=casa::ROMSPolarizationColumns(this->ms->polarization()).numCorr().get(0);
    //We retrieve polarization type
    casa::Vector<casa::String> polType=casa::ROMSFeedColumns(this->ms->feed()).polarizationType()(0);
    if ((polType[0]=="L")||(polType[0]=="R"))
         this->polType=mtimager::PolarizationType::Circular;
    else if ((polType[0]=="X")||(polType[0]=="Y"))
        this->polType=mtimager::PolarizationType::Linear;
    else if (this->noOfPolarizations==1) this->polType=mtimager::PolarizationType::Linear; //Not really important
    else throw ImagerException("Unable to retrieve polarisation");
    
    ROMSSpWindowColumns spectralWindows(this->ms->spectralWindow());
    casa::Array<casa::MFrequency> freq;
    measFreq.clear();
    if (spectralWindows.nrow()!=1)
    {
        throw ImagerException("Only MS files with one spectral Window entries are currently supported");
    }
    spectralWindows.chanFreqMeas().get(0,freq,true);
    if (freq.ndim()!=1) throw ImagerException ("Unexpected dimension of returned array from MS file");
    for (unsigned int i=0; i<freq.nelements();i++)
    {   
           this->measFreq.push_back(freq(IPosition(1,i)));
    }
    
    if (this->ms->observation().nrow()!=1)
        throw ImagerException("Currently MS files with 1 observations are supported");
    this->telescopeName=ROMSObservationColumns(this->ms->observation()).telescopeName().get(0);
    if (!casa::MeasTable::Observatory(this->telescopePosition,casa::String(this->telescopeName)))
        throw ImagerException("Query to casa::MeasTable::Observatory() for telescope position was unsuccessful");
    
    //We need to have th sort index initiated.... as to avaid complicating checking for getNoOfRecords().. etc
    int noOfAntennas=this->ms->antenna().nrow();
    this->sortIndex=new SortIndex(this->objectName+".IndexCreater",this->data_antenna1,this->data_antenna2,noOfAntennas,this->statisticManager);
    
    this->asyncLoadingInit=false;
    //The below created thread will not do anything before the above variable is set to true;
    this->myAsyncLoadingThread=new boost::thread(ThreadEntry<MSVisibilityDataSetAsync>(this,&MSVisibilityDataSetAsync::asyncCasaArrayLoad));
}

MSVisibilityDataSetAsync::~MSVisibilityDataSetAsync()
{
    //TO DO
}
int MSVisibilityDataSetAsync::getNoOfChunks()
{
    if (this->noOfChunks!=-1) return this->noOfChunks;
    int noOfRecords=this->getNoOfRecords();
    int chunks=noOfRecords/this->chunkLength;
    if (noOfRecords%this->chunkLength)
        chunks++;
    this->noOfChunks=chunks;
    return chunks;
}
int MSVisibilityDataSetAsync::getNoOfRecords()
{
    return this->sortIndex->getNoOfRecords();
}
enum PolarizationType::Type MSVisibilityDataSetAsync::getPolarizationType()
{
    return this->polType;
}
void MSVisibilityDataSetAsync::setChannelChoice(std::vector<bool>& channelChoice)
{
    //before setting ensure proper set up... and we can initiate all system... but no loading
    if (channelChoice.size()!=this->measFreq.size())
        throw ImagerException("Channel Choice vector is not consistent");
    this->channelChoice=channelChoice;
    
    this->initDataManagers(); 
}
void MSVisibilityDataSetAsync::initDataManagers()
{
    int noOfChannels=this->measFreq.size();
  
    this->uvwManager=new UVWStorageManager(this->getFactory(),this->objectName+".uvw","uvw",this->statisticManager,*this->sortIndex,3, 3,this->chunkLength,this->parallel_channels);
    this->weightsManager=new WeightsStorageManager(this->getFactory(),this->objectName+".weights","weights",this->statisticManager,*this->sortIndex,this->noOfPolarizations,this->noOfPolarizations,this->chunkLength,this->parallel_channels);
    //this->timeSorted=new TimeStorageManager(this->objectName+".time","time",this->statisticManager,*this->sortIndex,1,1,this->chunkLength,this->parallel_channels);
    if (this->doPSF)
    {
        this->visibilityForPSFManager=new VisibilityForPSFStorageManager(this->getFactory(),this->objectName+".PSFVisibilty",string("visibility"),this->statisticManager,*this->sortIndex,this->noOfPolarizations,
                        this->chunkLength);
        this->visibilityManager=NULL;
    }
    else
    {
        this->visibilityManager=new VisibilityStorageManager(this->getFactory(),this->objectName+".Visibilty",string("visibility"),this->statisticManager,*this->sortIndex,this->noOfPolarizations,
                        noOfChannels*this->noOfPolarizations, this->chunkLength,this->parallel_channels);
        this->visibilityForPSFManager=NULL;
    }
    this->flagsManager=new FlagsStorageManager(this->getFactory(),this->objectName+".flags",string("flags"),this->statisticManager,*this->sortIndex,this->noOfPolarizations, 
                    noOfChannels*this->noOfPolarizations, this->chunkLength,this->parallel_channels);
    this->frequencyManager=new FrequencyConverter(this->getFactory(),this->objectName+".LSRKfreqeuncy",this->statisticManager,this->measFreq,this->channelChoice,this->chunkLength,*this->sortIndex,this->phaseCentre, this->telescopePosition,this->parallel_channels,this->epochType);
    this->uvwManager->setNextChannelStorage(0);
    this->weightsManager->setNextChannelStorage(0);
  //  this->timeSorted->setNextChannelStorage(0);
    this->uvwManager->init_sorting();
    this->weightsManager->init_sorting();
    //this->timeSorted->init_sorting();
    for (int channelNo=0;channelNo<noOfChannels;channelNo++)
    {
      //  FrequencyConverter *freq=NULL;
        int offsetFlags=-1;
        int offsetVis=-1;
        if (this->channelChoice[channelNo])
        {
            std::stringstream channel;
            channel<<channelNo;
            
            offsetVis=channelNo*this->noOfPolarizations;
            offsetFlags=channelNo*this->noOfPolarizations;
        //    freq=new FrequencyConverter(this->objectName+".frequencyConverter",string("channel-")+channel.str(),this->statisticManager,this->measFreq[channelNo],*this->timeSorted,
          //          *this->sortIndex, this->chunkLength,this->phaseCentre,this->telescopePosition);
            
        }
        if (!this->doPSF)
        {
            this->visibilityManager->setNextChannelStorage(offsetVis);
        }
        this->flagsManager->setNextChannelStorage(offsetFlags);
      //  this->freqSortedVector.push_back(freq);
        
    }
        if (!this->doPSF)
            this->visibilityManager->init_sorting();
        else 
            this->visibilityForPSFManager->init_arrays();
    
    this->flagsManager->init_sorting();
    
}
    int MSVisibilityDataSetAsync::getNoOfRecordsInChunk(int chunkNo)
{
    int noOfChunks=getNoOfChunks();
    if (chunkNo>=noOfChunks) throw ImagerException("Out of bounds chunk No");
    int noOfRecords=this->chunkLength;
    if (chunkNo==(noOfChunks-1))
    {
        if (this->lastChunkLength==-1)
        {
            this->lastChunkLength=this->getNoOfRecords()%this->chunkLength;
            if (this->lastChunkLength==0) this->lastChunkLength=this->chunkLength;
        }
        noOfRecords=this->lastChunkLength;
    }
    return noOfRecords;
    
}
GAFW::Array * MSVisibilityDataSetAsync::getUVWArray(int chunkNo)
{
    {
        scoped_detailed_timer t((new DetailedTimerStatistic(this,"Wait For Chunk Ready","UVW",chunkNo)),this->statisticManager);
        return this->uvwManager->getArray(0,chunkNo);
    }

}
GAFW::Array * MSVisibilityDataSetAsync::getVisibilitiesArray(int chunkNo, int channelNo)
{
    if (channelNo==-1) return NULL; //There is nothing to do
    if (!this->doPSF)
    {
        scoped_detailed_timer t((new DetailedTimerStatistic(this,"Wait For Chunk Ready","Visibilities",chunkNo)),this->statisticManager);
        return this->visibilityManager->getArray(channelNo,chunkNo);
    }
    else
    {
        scoped_detailed_timer t((new DetailedTimerStatistic(this,"Wait For Chunk Ready","PSFVisibilities",chunkNo)),this->statisticManager);
        return this->visibilityForPSFManager->getArray(chunkNo);
    }
}
GAFW::Array * MSVisibilityDataSetAsync::getFlagsArray(int chunkNo, int channelNo)
{
     if (channelNo==-1) return NULL;
    {
        scoped_detailed_timer t((new DetailedTimerStatistic(this,"Wait For Chunk Ready","Flags",chunkNo)),this->statisticManager);
        return this->flagsManager->getArray(channelNo,chunkNo);
    }
 
}
GAFW::Array * MSVisibilityDataSetAsync::getFrequencyArray(int chunkNo, int channelNo)
{
    if (channelNo==-1) return NULL;
    {
        scoped_detailed_timer t((new DetailedTimerStatistic(this,"Wait For Chunk Ready","Frequency",chunkNo)),this->statisticManager);
        return this->frequencyManager->getArray(channelNo,chunkNo);
    }
    
}
GAFW::Array * MSVisibilityDataSetAsync::getWeightsArray(int chunkNo, int channelNo)
{
    if (channelNo==-1) return NULL; ///don't fill anything if we have a run over a channel 
    {
        scoped_detailed_timer t((new DetailedTimerStatistic(this,"Wait For Chunk Ready","Weights",chunkNo)),this->statisticManager);
        return this->weightsManager->getArray(0,chunkNo);
    }
    
}
GAFW::Array * MSVisibilityDataSetAsync::getArray(enum VisData::DataType type, int chunkNo,int channelNo)
{
    switch (type)
    {
        case VisData::UVW:
            return this->getUVWArray(chunkNo);
            break;
        case VisData::WEIGHT:
            return this->getWeightsArray(chunkNo,channelNo);
            break;
        case VisData::FREQUENCY:
            return this->getFrequencyArray(chunkNo,channelNo);
            break;
        case VisData::FLAGS:
            return this->getFlagsArray(chunkNo,channelNo);
            break;
        case VisData::VISIBILITY:
            
            return this->getVisibilitiesArray(chunkNo,channelNo);
            break;
        default:
            throw ImagerException("The data requested for load is not known");
    }
}

void MSVisibilityDataSetAsync::initAsyncLoading()
{
    boost::mutex::scoped_lock lock(this->myAsyncLoadingMutex);
    this->asyncLoadingInit=true;
    this->myAsyncLoadingCond.notify_all();
}
void MSVisibilityDataSetAsync::asyncCasaArrayLoad()
{
    boost::mutex::scoped_lock lock(this->myAsyncLoadingMutex);
    
    while(!this->asyncLoadingInit)
        this->myAsyncLoadingCond.wait(lock);
    //Ok...all we have to do invoke a load of each column
    {
        scoped_detailed_timer t((new DetailedTimerStatistic(this,"Load from MS","Antenna1",-1)),this->statisticManager);
        this->data_antenna1.loadData(this->mainCols->antenna1());
    }
    {
        scoped_detailed_timer t((new DetailedTimerStatistic(this,"Load from MS","Antenna2",-1)),this->statisticManager);
        this->data_antenna2.loadData(this->mainCols->antenna2());
    }
        //Automatiocally sorting should begin
    {
        scoped_detailed_timer t((new DetailedTimerStatistic(this,"Load from MS","Time",-1)),this->statisticManager);
        this->frequencyManager->loadData(this->mainCols->time());
    }
    {   
        scoped_detailed_timer t((new DetailedTimerStatistic(this,"Load from MS","UVW",-1)),this->statisticManager);
        this->uvwManager->columnLoader.loadData(this->mainCols->uvw());
        
        
    }
    
    {
        scoped_detailed_timer t((new DetailedTimerStatistic(this,"Load from MS","Weight",-1)),this->statisticManager);
        this->weightsManager->columnLoader.loadData(this->mainCols->weight());
    }
    if (!this->doPSF)
    {
        scoped_detailed_timer t((new DetailedTimerStatistic(this,"Load from MS","Visibility",-1)),this->statisticManager);
        this->visibilityManager->columnLoader.loadData(this->mainCols->data());
    }
    {
        scoped_detailed_timer t((new DetailedTimerStatistic(this,"Load from MS","Flags",-1)),this->statisticManager);
        this->flagsManager->columnLoader.loadData(this->mainCols->flag());
    }
    
    //And all is done
}

