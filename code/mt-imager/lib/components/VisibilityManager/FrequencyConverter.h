/* FrequencyConverter.h: Definition of the FrequencyConverter class. 
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


#ifndef FREQUENCYCONVERTER_H
#define	FREQUENCYCONVERTER_H
#include "MSVisibilityDataSetAsync.h"
#include "measures/Measures.h"
#include <map>
namespace mtimager
{
    class FrequencyConverter: public GAFW::Identity, public GAFW::FactoryAccess {
    private:
        FrequencyConverter(const FrequencyConverter& orig){};
       
    protected:
        typedef SortedStorage<float,float,GAFW::GeneralImplimentation::real_float,CopyStyle::Normal> FrequencySortedStorage;
        FrequencySortedStorage ** sortedStorage;
        casa::MDirection phaseCentre;
        casa::MPosition telescopePosition;
        double * measuredFreqs; 
        int noOfChannels;
        casa::MEpoch::Types timeType;
        std::vector<int> choosenChannelNos;
        std::vector<boost::thread *> threads;
        GAFW::FactoryStatisticsOutput *statisticManager;
         int parallel_calc;
        
        float ** stores;
        boost::thread *myThread;
        MSVectorColumnLoader<double> raw_time;
        
        double *time_storage;
        int noOfRecords;
        void asyncFrequencyCalcThread();
        void calculate(float **freqstores,double * freq,int noOfStores);
        void sortThreadFunc(int offset);
    public:
        
       // float * getStoragePointer(int channelNo);
        FrequencyConverter(GAFW::Factory * factory,std::string objectName, GAFW::FactoryStatisticsOutput *statisticManager,std::vector<casa::MFrequency> frequencies,std::vector<bool> choosenChannels,int chunksLength,SortIndex &index,casa::MDirection phaseCentre, casa::MPosition telescopePosition,int parallel_calc,casa::MEpoch::Types timeType);
        virtual ~FrequencyConverter();
        GAFW::Array * getArray(int channelNo,int chunkNo);
        void loadData(const casa::ROScalarColumn<double> &timecol);
    protected:
        class MyThreadEntry
              {
              protected:
                  int offset;
                  FrequencyConverter &conv;
              public:
                  MyThreadEntry(FrequencyConverter &conv,int offset):offset(offset),conv(conv){}
                  void operator()() 
                  {
                      conv.sortThreadFunc(offset);
                  }
              };

    };
}

#endif	/* FREQUENCYCONVERTER_H */

