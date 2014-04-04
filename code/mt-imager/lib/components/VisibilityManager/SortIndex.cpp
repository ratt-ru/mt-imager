/* SortIndex.cpp: Implementation  of the SortIndex class. 
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


#include "SortIndex.h"
#include "statistics/scoped_timer.h"
#include "statistics/DetailedTimeStatistic.h"

#include "MTImagerException.h"
using namespace mtimager;

        
SortIndex::SortIndex(std::string nickname,MSVectorColumnLoader<int> &antenna1, MSVectorColumnLoader<int> &antenna2, const int noOfAntennas,GAFW::FactoryStatisticsOutput *statisticManager):
        antenna1(antenna1),antenna2(antenna2),noOfAntennas(noOfAntennas),Identity(nickname,"SortIndex")
{
    this->statisticManager=statisticManager;
    this->sortIndex=NULL;
    this->sortReady=false;
    this->sortThread= new boost::thread(ThreadEntry<SortIndex>(this,&SortIndex::sortFuncThread));

}
SortIndex::~SortIndex()
{
    this->sortThread->join();
    if (this->sortIndex!=NULL)
    delete[] this->sortIndex;
    
    
}
void SortIndex::sortFuncThread()
{
    
        const int * antenna1_data=this->antenna1.getStorage();
        const int * antenna2_data=this->antenna2.getStorage();
        this->noOfRecords=this->antenna1.nelements();
        scoped_timer<DetailedTimerStatistic> a(new DetailedTimerStatistic(this,"Index Creation", "N/A",-1),this->statisticManager);
        std::vector<int> * index_list=new std::vector<int>[this->noOfAntennas*this->noOfAntennas];
        int initialNoOfRecords=this->noOfRecords;
        for (int i=0;i<initialNoOfRecords;i++)
        {
            if (antenna1_data[i]==antenna2_data[i]) 
            {   
                this->noOfRecords--;
                continue;
            }
            
            int baselineNo=(this->noOfAntennas*this->noOfAntennas)-antenna1_data[i]*noOfAntennas-antenna2_data[i]-1;
            index_list[baselineNo].push_back(i);
            
        }
        this->sortIndex=new int[this->noOfRecords];
        int indexNo=0;
        for (int baseline=0;baseline<noOfAntennas*noOfAntennas;baseline++)
        {
            for (std::vector<int>::iterator i=index_list[baseline].begin();i!=index_list[baseline].end();i++)
            {
                this->sortIndex[indexNo]=*i;
                indexNo++;
            }
        
        }
        
        if (indexNo!=this->noOfRecords) throw ImagerException("indexNo Value unexpected");
        delete[] index_list;
        {
            boost::mutex::scoped_lock lock(this->myMutex);
            this->sortReady=true;
            this->myCond.notify_all();
        }
        //Thread will end here
}
const int * SortIndex::getSortIndex()
{
    this->wait_for_sort_ready();
    return this->sortIndex;
}

const int SortIndex::getNoOfRecords()
{
    this->wait_for_sort_ready();
    return this->noOfRecords;
}
void SortIndex::wait_for_sort_ready()
{
   
    boost::mutex::scoped_lock lock(this->myMutex);
    while(!this->sortReady)
    {
        this->myCond.wait(lock);
    }
}
        

