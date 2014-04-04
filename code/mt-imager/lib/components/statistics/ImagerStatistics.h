/* ImagerStatistics.h:  Definition of the ImagerStatistics imager component and class. 
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

#ifndef IMAGERSTATISTICS_H
#define	IMAGERSTATISTICS_H
#include "gafw.h"
#include "WImager.h"
#include "tools/CSVWriter/CSVWriter.h"
#include "GPUafw.h"
#include "ImagerTimeStatistic.h"
#include "DetailedTimeStatistic.h"
#include <boost/thread.hpp>
typedef GAFW::GPU::SynchronousQueue<GAFW::Statistic *> StatisticsQueue;
namespace mtimager { namespace statistics {
    class ImagerStatistics : public GAFW::FactoryStatisticsOutput, public GAFW::LogFacility, public GAFW::Identity {
        ImagerStatistics(const ImagerStatistics& orig){};
    public:
        struct Conf
        {
            std::string mainStatisticsFile;
            std::string griddingStatisticsFile;
            std::string engineStatisticsFile;
            int nx;
            int support;
            int sampling;
            float inc;
        };
    protected:
        GAFW::Tools::CSVWriter::CSVWriter *imagerTimerPerformance;
        GAFW::Tools::CSVWriter::CSVColumn<long long int> cycleNoCol;
        GAFW::Tools::CSVWriter::CSVColumn<std::string> actionTypeCol;
        GAFW::Tools::CSVWriter::CSVColumn<std::string> actionDetailCol;
        GAFW::Tools::CSVWriter::CSVColumn<std::string> identity_nameCol;
        GAFW::Tools::CSVWriter::CSVColumn<std::string> identity_nickNameCol;
        GAFW::Tools::CSVWriter::CSVColumn<double> startCol;
        GAFW::Tools::CSVWriter::CSVColumn<double> endCol;
        GAFW::Tools::CSVWriter::CSVColumn<double> duration;
        boost::chrono::high_resolution_clock::time_point startTimePoint;
        
        //Following are for GPUEngineStatistics
        GAFW::Tools::CSVWriter::CSVWriter *engineStatistics;
        GAFW::Tools::CSVWriter::CSVColumn<long long int> snapshotNo;
        GAFW::Tools::CSVWriter::CSVColumn<std::string> operatorNickname;
        GAFW::Tools::CSVWriter::CSVColumn<std::string> operatorName;
        GAFW::Tools::CSVWriter::CSVColumn<float> kernelDuration;
        
        
        GAFW::Tools::CSVWriter::CSVWriter *griddingStatistics;
        GAFW::Tools::CSVWriter::CSVColumn<int> griddedRecords;
        GAFW::Tools::CSVWriter::CSVColumn<int> inputRecords;
        GAFW::Tools::CSVWriter::CSVColumn<int> compressedRecords;
        GAFW::Tools::CSVWriter::CSVColumn<long int> totalGriddedPoints;
        GAFW::Tools::CSVWriter::CSVColumn<long int> totalCompressedPoints;
        GAFW::Tools::CSVWriter::CSVColumn<float> averageGriddedSupport;
        GAFW::Tools::CSVWriter::CSVColumn<float> averageCompressedSupport;
        GAFW::Tools::CSVWriter::CSVColumn<float> gridPointsRealRate;
        GAFW::Tools::CSVWriter::CSVColumn<float> gridRecordsRealRate;
        GAFW::Tools::CSVWriter::CSVColumn<float> gridPointsTotalRate;
        GAFW::Tools::CSVWriter::CSVColumn<float> gridRecordsTotalRate;
        GAFW::Tools::CSVWriter::CSVColumn<float> phase1Time;
        GAFW::Tools::CSVWriter::CSVColumn<float> phase1Rate;
        
        long int acu_totalGriddedPoints;
        long int acu_totalCompressedPoints;
        long int acu_totalGriddedRecords;
        long int acu_totalCompressedrecords;
        float acu_kernel_exec;
        long int acu_total_records;
        float acu_total_phase1;
        int polarizations;
        
        
        
        std::map<int,GAFW::GPU::GPUEngineOperatorStatistic*> griddingEngineStat;
        std::map<int,mtimager::WImager::WImagerStatistic *> griddingWImagerStat;
        std::map<int,double> phase1time;
        
        
        
        struct Conf conf;
        boost::thread * mythread;
        StatisticsQueue *myQueue;
    public:
        
        ImagerStatistics(Conf conf);
        virtual void push_statistic(GAFW::Statistic *);
        void statisticsThreadFunc();
        void handleTimeStatistic(ImagerTimeStatistic::Data *stat);
        void handleDetailedTimerStatistic(DetailedTimerStatistic *stat);
        void handleWImagerStatistic(mtimager::WImager::WImagerStatistic *stat);
        void handleGPUEngineOperatorStatistic(GAFW::GPU::GPUEngineOperatorStatistic*stat);
        void hangleCompleteGriddingStatistic(GAFW::GPU::GPUEngineOperatorStatistic*stat,mtimager::WImager::WImagerStatistic *stat2);
        virtual ~ImagerStatistics();
        void finalize();
    

    };
}};
#endif	/* IMAGERSTATISTICS_H */

