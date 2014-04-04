/* MSVisibilityDataSetAsync.h: Definition  of the MSVisibilityDataSetAsync  class. 
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
#ifndef MSVISIBILITYDATASETASYNC_H
#define	MSVISIBILITYDATASETASYNC_H
#include "gafw.h"
#include "SortedStorage.h"
#include "StorageManager.h"
#include "EpochColumnLoader.h"
#include "MSColumnLoader.h"
#include "FrequencyConverter.h"
#include "VisibilityForPSFStorageManager.h"
#include "VisibilityManager.h"
#include "ms/MeasurementSets/MeasurementSet.h"
#include "ms/MeasurementSets/MSMainColumns.h"
#include "measures/Measures.h"
#include "ms/MeasurementSets.h"
#include "measures/Measures/MCFrequency.h"
#include "measures/Measures/MeasTable.h"
#include "measures/TableMeasures.h"
//#include "VisibilityDataSet.h"
#include "ImagerTimeStatistic.h"
#include <boost/thread.hpp>
namespace mtimager
{   
    
    
    //class VisibilityManager { struct VisData {enum DataType;}; };;
    //struct VisibilityManager::VisData;
    class MSVisibilityDataSetAsync:virtual public GAFW::Identity,virtual public GAFW::LogFacility, virtual public GAFW::FactoryAccess {
    protected:
        enum mtimager::PolarizationType::Type polType;
        int parallel_channels;
        casa::MeasurementSet *ms;
        int noOfPolarizations;
        int noOfPolarizationsTypes;
        int chunkLength;
        int lastChunkLength;
        int noOfChunks;
        bool doPSF;
        std::vector<bool> channelChoice;
        std::vector<casa::MFrequency> measFreq;
        std::string telescopeName;
        casa::MDirection phaseCentre;
        casa::MPosition telescopePosition;
        SortIndex * sortIndex;
        casa::MEpoch::Types epochType;
        casa::ROMSMainColumns *mainCols;
        MSVectorColumnLoader<int> data_antenna1;
        MSVectorColumnLoader<int> data_antenna2;
        MSVectorColumnLoader<double> data_time;
        UVWStorageManager *uvwManager;
        VisibilityStorageManager *visibilityManager;
        FlagsStorageManager *flagsManager;
        WeightsStorageManager *weightsManager;
        VisibilityForPSFStorageManager *visibilityForPSFManager;
        FrequencyConverter *frequencyManager;
        //std::vector<FrequencyConverter *> freqSortedVector;
        GAFW::FactoryStatisticsOutput *statisticManager;
        
        bool asyncLoadingInit;
        boost::condition_variable myAsyncLoadingCond;
        boost::mutex myAsyncLoadingMutex;
        boost::thread * myAsyncLoadingThread;
        void asyncCasaArrayLoad();
        void initSortStorages();
        void initDataManagers();
        
        int getNoOfRecordsInChunk(int chunkNo);
        static std::string convertDots(std::string str);
        casa::Table *tab;
        casa::Table *tabselect;
    private:
        MSVisibilityDataSetAsync(const MSVisibilityDataSetAsync& orig){};
    public:
        MSVisibilityDataSetAsync(GAFW::Factory *factory, std::string msFileName,GAFW::FactoryStatisticsOutput *statisticManager,int chunkLength,int paralel_channels,int field,bool doPSF=false);
        virtual ~MSVisibilityDataSetAsync(); //All buffers are cleared on delete
        inline int getNoOfPolarizations();
        inline int getNoOfPolarizationsTypes();
        inline std::vector<casa::MFrequency>  getChannelFrequencies();
        inline std::string getTelescopeName();
        inline casa::MDirection getPhaseCentre();
        int getNoOfChunks();
        int getNoOfRecords();
        void setChannelChoice(std::vector<bool>& channelChoice);
        GAFW::Array * getUVWArray(int chunkNo);
        GAFW::Array * getVisibilitiesArray(int chunkNo, int channelNo);
        GAFW::Array * getFlagsArray(int chunkNo, int channelNo);
        GAFW::Array * getFrequencyArray(int chunkNo, int channelNo);
        GAFW::Array * getWeightsArray(int chunkNo, int channelNo);
        GAFW::Array * getArray(enum VisData::DataType type, int chunkNo,int channelNo);
        
       
        void initAsyncLoading();
        enum PolarizationType::Type getPolarizationType();
    };
    int MSVisibilityDataSetAsync::getNoOfPolarizations()
    {
        return this->noOfPolarizations;
    }
    int MSVisibilityDataSetAsync::getNoOfPolarizationsTypes()
    {
        return this->noOfPolarizationsTypes;
    }
    std::vector<casa::MFrequency> MSVisibilityDataSetAsync::getChannelFrequencies()
    {
        return this->measFreq;
    }
    std::string MSVisibilityDataSetAsync::getTelescopeName()
    {
        return this->telescopeName;
    }
    casa::MDirection MSVisibilityDataSetAsync::getPhaseCentre()
    {
        return this->phaseCentre;
    }

}

#endif	/* MSVISIBILITYDATASETASYNC_H */

