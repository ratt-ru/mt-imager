/* FrequencyConverter.cpp: Implementation of the FrequencyConverter class. 
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
#include <iostream>
#include "FrequencyConverter.h"
using namespace mtimager;

FrequencyConverter::FrequencyConverter(GAFW::Factory * factory,std::string objectName, GAFW::FactoryStatisticsOutput *statisticManager,std::vector<casa::MFrequency> frequencies,std::vector<bool> choosenChannels,int chunksLength,SortIndex &index,casa::MDirection phaseCentre, casa::MPosition telescopePosition,int parallel_calc,casa::MEpoch::Types timeType)
        :Identity(objectName,"FrequencyConverter"),FactoryAccess(factory),statisticManager(statisticManager),parallel_calc(parallel_calc),timeType(timeType)
{
    FactoryAccess::init();
       
    this->telescopePosition=telescopePosition;
    this->phaseCentre=phaseCentre;
    this->noOfChannels=frequencies.size();
    this->measuredFreqs=new double[this->noOfChannels];
    this->sortedStorage=new FrequencySortedStorage*[this->noOfChannels];
    for (int channelNo=0;channelNo<this->noOfChannels;channelNo++)
    {
        std::stringstream ch;
        ch <<channelNo;
        
        if (choosenChannels[channelNo])
        {
            this->measuredFreqs[channelNo]=frequencies[channelNo].getValue().getValue();
            this->sortedStorage[channelNo]=new FrequencySortedStorage(this->getFactory(),this->objectName+ch.str(),"Frequency",this->statisticManager,index,1, 0,1,chunksLength);
            this->choosenChannelNos.push_back(channelNo);
        }
        else
        {
            this->measuredFreqs[channelNo]=-1.0;
            this->sortedStorage[channelNo]=NULL;
        }
    }
       this->stores=NULL;
       
       this->getFactory()->registerIdentity(this);
       
       this->myThread=new boost::thread(ThreadEntry<FrequencyConverter>(this,&FrequencyConverter::asyncFrequencyCalcThread));
        
    
    
}
FrequencyConverter::~FrequencyConverter()
{
    this->myThread->join();
    if (this->stores!=NULL)
    {
        for (int i=0;i<this->noOfChannels;i++)
        {
            if (this->stores[i]!=NULL)
            {
                delete [] this->stores[i];
            }
                    
        }
        delete [] this->stores;
    }
       
}

void FrequencyConverter::asyncFrequencyCalcThread()
{
    
    this->time_storage=this->raw_time.getStorage();
    this->noOfRecords=this->raw_time.nelements();
    //Ok let;s crete all storages
    this->stores=new float*[this->noOfChannels];
    for (int channelNo=0;channelNo<this->noOfChannels;channelNo++)
    {
        if (this->measuredFreqs[channelNo]>0.0) this->stores[channelNo]=new float[this->noOfRecords];
        else
            this->stores[channelNo]=NULL;
    }
    int noOfChosenChannels=this->choosenChannelNos.size();
    float ** currentStores=new float*[noOfChosenChannels];
    double *currentFreq=new double[noOfChosenChannels];
    for (int channelNo=0;channelNo<noOfChosenChannels;channelNo++)
    {
        int properChannel=this->choosenChannelNos[channelNo];
         currentFreq[channelNo]=this->measuredFreqs[properChannel];
         currentStores[channelNo]=this->stores[properChannel];
    }
    {
        scoped_detailed_timer t((new DetailedTimerStatistic(this,"Conversion of Frequency to LSRK","N/A",-1)),this->statisticManager);
        this->calculate(currentStores,currentFreq,noOfChosenChannels);
    }
    //Now we initiate sorting threads
    for (int i=0;i<this->parallel_calc;i++)
    {
        this->threads.push_back(new boost::thread(MyThreadEntry(*this,i)));
    }
    //An we are ready
    
}

void FrequencyConverter::calculate(float **freqstores,double *measFreq,int noOfStores)
{
    double  last_time=0;
    float last_answers[noOfStores];
    double c=299792458.0;
    for (int recordNo=0;recordNo<this->noOfRecords;recordNo++)
    {
        if (this->time_storage[recordNo]==last_time)
        {
                for (int freqNo=0;freqNo<noOfStores;freqNo++)
                {
                        freqstores[freqNo][recordNo]=last_answers[freqNo];
                }
        }
        else
        {
            casa::MeasFrame frame(casa::MEpoch(casa::Quantity(this->time_storage[recordNo],"s"),this->timeType),this->telescopePosition,this->phaseCentre);
            casa::MFrequency::Convert tolsr((casa::MFrequency::Types) casa::MFrequency::TOPO,
            casa::MFrequency::Ref(casa::MFrequency::LSRK,frame));
            for (int freqNo=0;freqNo<noOfStores;freqNo++)
            {
                    freqstores[freqNo][recordNo]=(float)(tolsr(measFreq[freqNo]).getValue().getValue()/c);
                    last_time=this->time_storage[recordNo];
                    last_answers[freqNo]=freqstores[freqNo][recordNo];
            }
            
        }
    }
    
}
GAFW::Array * FrequencyConverter::getArray(int channelNo,int chunkNo)
{
    return this->sortedStorage[channelNo]->getArray(chunkNo);
}
void FrequencyConverter::loadData(const casa::ROScalarColumn<double> &timecol)
{
    this->raw_time.loadData(timecol);
}
void FrequencyConverter::sortThreadFunc(int offset)
{
    for (int chosenChannelNo=offset;chosenChannelNo<(int)this->choosenChannelNos.size();chosenChannelNo+=this->parallel_calc)
    {
//        int ss=this->choosenChannelNos.size();
        int properChannelNo=this->choosenChannelNos[chosenChannelNo];
        this->sortedStorage[properChannelNo]->createArrays();
    }
    for (int chosenChannelNo=offset;chosenChannelNo<(int)this->choosenChannelNos.size();chosenChannelNo+=this->parallel_calc)
    {
//        int ss=this->choosenChannelNos.size();
        int properChannelNo=this->choosenChannelNos[chosenChannelNo];
        this->sortedStorage[properChannelNo]->sort(this->stores[properChannelNo]);
    }
}
 

