/* VisibilityManager.h: Definition  of the VisibilityManager component and class. 
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
#ifndef __VISIBILITYMANAGER_H__
#define	__VISIBILITYMANAGER_H__
#include <string>
#include <vector>
#include "mtimager.h" //temporary
namespace mtimager
{
    class MSVisibilityDataSetAsync;
   class VisibilityManager: public  GAFW::FactoryAccess, public GAFW::LogFacility, public GAFW::Identity {
    public:
        
        struct Conf
        {
            GAFW::FactoryStatisticsOutput *statisticsSystem;
            int maxRecords;
            std::vector<int> channels;
            bool allChannels;
            std::string dataType;
            std::vector<std::string> dataFileNames;
            int parallel_channels;
            int field;
            bool doPSF;
        };
    protected:
        struct Conf conf;
        std::vector<MSVisibilityDataSetAsync *> dataSets;
        std::vector<int> channels;  //first entry is always -1.... for logistical reasons
        std::vector<std::pair<int,int> > currentChannelGroup;  //first point to the proper channel No, second points to the Image No  
        std::vector<std::pair<int,int> >::iterator currentChannel;
       
        string telescopeName;
        casa::MPosition telescopePosition;
        casa::MDirection phaseCentre;
        bool outputValid;
        bool topoFrequencyConstant;
        int noOfPolarizations;
        int currentChunkNo;
        int currentDataSet;
        int channelNoForNextGroup;
        enum PolarizationType::Type polType;
        std::vector<casa::MFrequency> measChannelFreqs;
        std::vector<casa::MFrequency> measChosenChannelFreqs;
        GAFW::ProxyResult *outputResult[VisData::TotalOutputs];
   public:
        
       
        
        VisibilityManager(GAFW::Factory *factory,std::string nickname, Conf conf);
        ~VisibilityManager();
       bool nextChannelGroup();
       bool nextChunk();
       GAFW::Result *getOutput(enum mtimager::VisData::DataType);
       std::vector<casa::MFrequency> getChannelFrequencies();
       std::vector<casa::MFrequency> getChoosenChannelFrequencies();
       int getNoOfPolarizations();
       casa::MDirection getPhaseCentre();  //temporary
       std::string getTelescopeName();
       enum PolarizationType::Type getPolarizationType();
       int nextChannel();
       std::vector<int> getCurrentChannels();
   };
   ;
}


#endif	/* VISIBILITYMANAGER_H */

