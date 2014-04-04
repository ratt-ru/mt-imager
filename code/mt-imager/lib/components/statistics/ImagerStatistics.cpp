/* ImagerStatistics.cpp:  Implementation of the ImagerStatistics imager component and class. 
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

#include "ImagerStatistics.h"
#include "ImagerTimeStatistic.h"
#include "WImager.h"
using namespace mtimager::statistics;
using namespace GAFW;

typedef boost::chrono::duration<double,boost::ratio<1,1000> > duration_ms;
ImagerStatistics::ImagerStatistics(Conf conf):Identity("ImagerStatistics","ImagerStastics"),
        startTimePoint(boost::chrono::high_resolution_clock::now()),
        cycleNoCol("Cycle No"),
        actionTypeCol("Type"),actionDetailCol("Detail"),
        identity_nameCol("Module Name"),identity_nickNameCol("Module Nickname"),
        startCol("Start"),endCol("End"), duration("Duration"), snapshotNo("Snapshot No"),
        operatorNickname("Operator Nickname"),operatorName("Operator Name"),kernelDuration("Kernel/s Execution Duration"),
        griddedRecords("Gridded Records"),
        compressedRecords("Compressed Records"),
        totalGriddedPoints("Total Gridded Points(Polarization factored in)"),
        totalCompressedPoints("Total Compressed Points (Polarization factored in)"),
        averageGriddedSupport("Average Gridded Support"),
        averageCompressedSupport("Average Compressed Support"),
        gridPointsRealRate("Real Gridding Points rate (Giga Grid Points/s)"),
        gridRecordsRealRate("Real Griding Records rate (Mega records/s)"),
        gridPointsTotalRate("Total Gridding Rate (Giga Grid Points/s)"),
        gridRecordsTotalRate("Total Gridding Records rate (Mega Records/s)")
        
{
    LogFacility::init();
    this->logDebug(execution_performance,"ImagerStatistics constructed");
    
    this->conf=conf;
    this->myQueue=new StatisticsQueue();
    this->logDebug(execution_performance,"Initiating thread");
    
    this->imagerTimerPerformance=new GAFW::Tools::CSVWriter::CSVWriter(this->conf.mainStatisticsFile.c_str());
    this->imagerTimerPerformance->addColumn(&this->cycleNoCol);
    this->imagerTimerPerformance->addColumn(&this->identity_nameCol);
    this->imagerTimerPerformance->addColumn(&this->identity_nickNameCol);
    this->imagerTimerPerformance->addColumn(&this->actionTypeCol);
    this->imagerTimerPerformance->addColumn(&this->actionDetailCol);
    
    this->imagerTimerPerformance->addColumn(&this->startCol);
    this->imagerTimerPerformance->addColumn(&this->endCol);
    this->imagerTimerPerformance->addColumn(&this->duration);
    this->imagerTimerPerformance->writeTitle();
    this->startCol.setFixed(false);
    this->endCol.setFixed(false);
    this->duration.setFixed(true);
    this->duration.setPrecision(100);
    
    this->engineStatistics=new GAFW::Tools::CSVWriter::CSVWriter(this->conf.engineStatisticsFile.c_str());
    this->engineStatistics->addColumn(&this->snapshotNo);
    this->engineStatistics->addColumn(&this->operatorNickname);
    this->engineStatistics->addColumn(&this->operatorName);
    this->engineStatistics->addColumn(&this->kernelDuration);
    this->engineStatistics->writeRow();
    this->griddingStatistics=new GAFW::Tools::CSVWriter::CSVWriter(this->conf.griddingStatisticsFile.c_str());
    this->griddingStatistics->addColumn(&this->snapshotNo);
    this->griddingStatistics->addColumn(&this->inputRecords);
    this->griddingStatistics->addColumn(&this->griddedRecords);
    this->griddingStatistics->addColumn(&this->compressedRecords);
    this->griddingStatistics->addColumn(&this->totalGriddedPoints);
    this->griddingStatistics->addColumn(&this->totalCompressedPoints);
    this->griddingStatistics->addColumn(&this->kernelDuration);
    this->griddingStatistics->addColumn(&this->averageGriddedSupport);
    this->averageGriddedSupport.setScientific(false);
    this->griddingStatistics->addColumn(&this->averageCompressedSupport);
    this->averageCompressedSupport.setScientific(false);
    this->griddingStatistics->addColumn(&this->gridPointsRealRate);
    this->gridPointsRealRate.setScientific(false);
    this->griddingStatistics->addColumn(&this->gridRecordsRealRate);
    this->gridRecordsRealRate.setScientific(false);
    this->griddingStatistics->addColumn(&this->gridPointsTotalRate);
    this->gridPointsTotalRate.setScientific(false);
    this->griddingStatistics->addColumn(&this->gridRecordsTotalRate);
    this->griddingStatistics->addColumn(&this->phase1Time);
    this->phase1Time.setScientific(false);
    this->phase1Time.setFixed(true);
    this->phase1Rate.setScientific(false);
    this->phase1Rate.setFixed(true);
    
    this->griddingStatistics->addColumn(&this->phase1Rate);
    
    this->gridRecordsTotalRate.setScientific(false);
    this->griddingStatistics->writeTitle();
    this->averageGriddedSupport.setFixed(5);
    this->averageCompressedSupport.setFixed(5);
    this->gridPointsRealRate.setFixed(5);
    this->gridRecordsRealRate.setFixed(5);
    this->gridPointsTotalRate.setFixed(5);
    this->gridRecordsTotalRate.setFixed(5);
    this->kernelDuration.setFixed(5);
    this->kernelDuration.setScientific(false);
    
    acu_totalGriddedPoints=0;
    acu_totalCompressedPoints=0;
    acu_totalGriddedRecords=0;
    acu_totalCompressedrecords=0;
    acu_kernel_exec=0.0f;
    this->acu_total_phase1=0.0f;
    this->acu_total_records=0;
    
    this->mythread=new boost::thread(GAFW::GPU::ThreadEntry<ImagerStatistics>(this,&ImagerStatistics::statisticsThreadFunc));
}
ImagerStatistics::~ImagerStatistics()
{
    {
  //      scoped_detailed_timer t((new DetailedTimerStatistic("Statistics finalization","N/A",-1)),this);
//        this->finalize();
    }
    GAFW::Statistic *null=NULL;
    this->myQueue->push(null);
    this->mythread->join();
   
    
}
void ImagerStatistics::finalize()
{

    this->snapshotNo.setValue(-1);
    this->griddedRecords.setValue(this->acu_totalGriddedRecords);
    this->inputRecords.setValue(this->acu_total_records);
    this->compressedRecords.setValue(this->acu_totalCompressedrecords);
    this->totalGriddedPoints.setValue(this->acu_totalGriddedPoints*this->polarizations);
    this->totalCompressedPoints.setValue(this->acu_totalCompressedPoints*this->polarizations);
    this->averageGriddedSupport.setValue(-1);
    this->averageCompressedSupport.setValue(-1);
    this->gridPointsRealRate.setValue(double(this->acu_totalGriddedPoints*this->polarizations)/(double(this->acu_kernel_exec)*1e6));
    this->gridRecordsRealRate.setValue(this->acu_totalGriddedRecords/(double(this->acu_kernel_exec)*1e3));
    this->gridPointsTotalRate.setValue(double(this->acu_totalGriddedPoints+this->acu_totalCompressedPoints)*double(4)/(double(this->acu_kernel_exec)*1e6));
    this->gridRecordsTotalRate.setValue(double(acu_totalCompressedrecords+this->acu_totalGriddedRecords)/(double(this->acu_kernel_exec)*1e3));
    this->kernelDuration.setValue(this->acu_kernel_exec);
    this->phase1Time.setValue(this->acu_total_phase1);
    this->phase1Rate.setValue((float)this->acu_total_records*1e-3/(this->acu_total_phase1));
   this->griddingStatistics->writeRow();
   /* std::cout <<"STATISTIC;"<< conf.nx <<';'<<conf.inc<<';'<<conf.support<<';'<<conf.sampling<<';'<<this->acu_total_records<<';'
            <<this->acu_totalGriddedRecords<<';'<<this->acu_totalCompressedrecords<<';'<<(float)this->acu_totalCompressedrecords/(float)this->acu_totalGriddedRecords<<';'
            <<this->acu_totalGriddedPoints*this->polarizations<<';'<<this->acu_totalCompressedPoints*this->polarizations<<';'<<(float)this->acu_totalCompressedPoints/(float)this->acu_totalGriddedPoints << ';'
            <<double(this->acu_totalGriddedPoints+this->acu_totalCompressedPoints)*double(this->polarizations)/(double(this->acu_kernel_exec)*1e6)<<';'<<double(this->acu_totalGriddedPoints*this->polarizations)/(double(this->acu_kernel_exec)*1e6)<<';'
            <<double(acu_totalCompressedrecords+this->acu_totalGriddedRecords)/(double(this->acu_kernel_exec)*1e3)<<';'<<this->acu_totalGriddedRecords/(double(this->acu_kernel_exec)*1e3)<<';'
           <<this->acu_total_phase1<< ';'<< (float)this->acu_total_records*1e-3/(this->acu_total_phase1)<<std::endl;
*/

}
    void ImagerStatistics::push_statistic(GAFW::Statistic *stat)
{
    this->myQueue->push(stat);
}

void ImagerStatistics::statisticsThreadFunc()
{
    this->logDebug(execution_performance,"Statistics Thread Initiated");
    for (;;)
    {
        
        Statistic * stat;
        this->myQueue->pop_wait(stat);
        if (stat==NULL)
        {	
		this->finalize();
            break; //Means that we are going to shutdown
        }
        if (dynamic_cast<ImagerTimeStatistic::Data*>(stat)!=NULL) handleTimeStatistic(dynamic_cast<ImagerTimeStatistic::Data*>(stat));
        else if (dynamic_cast<DetailedTimerStatistic*>(stat)!=NULL) handleDetailedTimerStatistic(dynamic_cast<DetailedTimerStatistic*>(stat));
        else if (dynamic_cast<WImager::WImagerStatistic*>(stat)!=NULL) handleWImagerStatistic(dynamic_cast<WImager::WImagerStatistic*>(stat));
        else if (dynamic_cast<GAFW::GPU::GPUEngineOperatorStatistic*>(stat)!=NULL) handleGPUEngineOperatorStatistic(dynamic_cast<GAFW::GPU::GPUEngineOperatorStatistic*>(stat));
        else this->logWarn(execution_performance,"An unknown statistic has been received");
        
    }
}

void ImagerStatistics::handleTimeStatistic(ImagerTimeStatistic::Data *stat)
{
    this->cycleNoCol.setValue(stat->cycleNo);
    this->identity_nameCol.setValue(stat->identity_name);
    this->identity_nickNameCol.setValue(stat->identity_objectName);
    this->actionTypeCol.setValue(stat->actionType);
    this->actionDetailCol.setValue(stat->actionDetail);
    duration_ms dur=stat->start_point-this->startTimePoint;
    double value=dur.count();
    this->startCol.setValue(value);
    dur=stat->end_point-this->startTimePoint;
    value=dur.count();
    this->endCol.setValue(value);
    dur=stat->end_point-stat->start_point;
    value=dur.count();
    this->duration.setValue(value);
    this->imagerTimerPerformance->writeRow();
}
void ImagerStatistics::handleDetailedTimerStatistic(DetailedTimerStatistic *stat)
{
    this->cycleNoCol.setValue(stat->cycleNo);
    this->identity_nameCol.setValue(stat->identity_name);
    this->identity_nickNameCol.setValue(stat->identity_objectName);
    this->actionTypeCol.setValue(stat->actionType);
    this->actionDetailCol.setValue(stat->actionDetail);
    duration_ms dur=stat->start_point-this->startTimePoint;
    double value=dur.count();
    this->startCol.setValue(value);
    dur=stat->end_point-this->startTimePoint;
    value=dur.count();
    this->endCol.setValue(value);
    dur=stat->end_point-stat->start_point;
    value=dur.count();
    this->duration.setValue(value);
    this->imagerTimerPerformance->writeRow();
}
void ImagerStatistics::handleGPUEngineOperatorStatistic(GAFW::GPU::GPUEngineOperatorStatistic* stat)
{
    this->operatorName.setValue(stat->operatorName);
    this->operatorNickname.setValue(stat->operatorNickname);
    this->snapshotNo.setValue(stat->snapshotNo);
    this->kernelDuration.setValue(stat->kernelExcecutionTime);
    this->engineStatistics->writeRow();
    if (this->phase1time.count(stat->snapshotNo)) this->phase1time[stat->snapshotNo]+=stat->kernelExcecutionTime;
    else
        this->phase1time[stat->snapshotNo]=stat->kernelExcecutionTime;
   
    if (stat->operatorName=="Operator::ConvGridder") 
    {
        this->acu_kernel_exec+=stat->kernelExcecutionTime;
        if ((this->phase1time[stat->snapshotNo]-stat->kernelExcecutionTime)>20.0f) 
        {
            this->phase1time[stat->snapshotNo]=10.0f+stat->kernelExcecutionTime;//not the right way but I cannot get over this problem 
           // std::cout << "WARNING ::: PHASE TIME HAD to BE controlled";
        }
        this->acu_total_phase1+=this->phase1time[stat->snapshotNo]-stat->kernelExcecutionTime;
        if (this->griddingWImagerStat.count(stat->snapshotNo))
        {
                this->hangleCompleteGriddingStatistic(stat,this->griddingWImagerStat[stat->snapshotNo]);
                this->griddingWImagerStat.erase(stat->snapshotNo);
        }
        else
        {
                this->griddingEngineStat[stat->snapshotNo]=stat;
        }
        
    }
    else
        delete stat;

    
}
void ImagerStatistics::handleWImagerStatistic(WImager::WImagerStatistic *stat)
{
    this->acu_totalCompressedPoints+=stat->compressedGridPoints;
    this->acu_totalCompressedrecords+=stat->compressedRecords;
    this->acu_totalGriddedPoints+=stat->griddedGridPoints;
    this->acu_totalGriddedRecords+=stat->griddedRecords;
    this->acu_total_records+=stat->noOfRecords;
        if (this->griddingEngineStat.count(stat->snapshotNo))
        {
                this->hangleCompleteGriddingStatistic(this->griddingEngineStat[stat->snapshotNo],stat);
                this->griddingEngineStat.erase(stat->snapshotNo);
        }
        else
        {
                this->griddingWImagerStat[stat->snapshotNo]=stat;
        }
    this->polarizations=stat->noOfPolarizations;
    
}
void ImagerStatistics::hangleCompleteGriddingStatistic(GAFW::GPU::GPUEngineOperatorStatistic*eng_stat,mtimager::WImager::WImagerStatistic *imager_stat)
{
    this->snapshotNo.setValue(eng_stat->snapshotNo);
    this->griddedRecords.setValue(imager_stat->griddedRecords);
    this->inputRecords.setValue(imager_stat->noOfRecords);
    this->compressedRecords.setValue(imager_stat->compressedRecords);
    this->totalGriddedPoints.setValue(imager_stat->griddedGridPoints*imager_stat->noOfPolarizations);
    this->totalCompressedPoints.setValue(imager_stat->compressedGridPoints*imager_stat->noOfPolarizations);
    this->averageGriddedSupport.setValue(sqrt((double)imager_stat->griddedGridPoints/(double)imager_stat->griddedRecords));
    this->averageCompressedSupport.setValue(sqrt((double)imager_stat->compressedGridPoints/(double)imager_stat->compressedRecords));
    this->gridPointsRealRate.setValue(double(imager_stat->griddedGridPoints)*double(imager_stat->noOfPolarizations)/(double(eng_stat->kernelExcecutionTime)*1e6));
    this->gridRecordsRealRate.setValue(double(imager_stat->griddedRecords)/(double(eng_stat->kernelExcecutionTime)*1e3));
    this->gridPointsTotalRate.setValue(double(imager_stat->griddedGridPoints+imager_stat->compressedGridPoints)*double(imager_stat->noOfPolarizations)/(double(eng_stat->kernelExcecutionTime)*1e6));
    this->gridRecordsTotalRate.setValue(double(imager_stat->griddedRecords+imager_stat->compressedRecords)/(double(eng_stat->kernelExcecutionTime)*1e3));
    this->kernelDuration.setValue(eng_stat->kernelExcecutionTime);
    this->phase1Time.setValue(this->phase1time[eng_stat->snapshotNo]-eng_stat->kernelExcecutionTime);
    this->phase1Rate.setValue(imager_stat->noOfRecords*1e-3/(this->phase1time[eng_stat->snapshotNo]-eng_stat->kernelExcecutionTime));
    this->griddingStatistics->writeRow();
    delete eng_stat;
    delete imager_stat;
}
