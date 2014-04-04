/* SortIndex.h: Definition  of the SortIndex  class. 
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
#ifndef __SORTINDEX_H__
#define	__SORTINDEX_H__
#include "MSColumnLoader.h"
#include <boost/thread.hpp>
#include "ThreadEntry.h"
#include "gafw.h"
namespace mtimager
{
    class SortIndex:public GAFW::Identity {
    private:
//        SortIndex(const SortIndex& orig){};
    protected:   
        MSVectorColumnLoader<int> &antenna1;
        MSVectorColumnLoader<int> &antenna2;
        const int noOfAntennas;
        int noOfRecords;
        bool sortReady;
        boost::mutex myMutex;
        boost::condition_variable myCond;
        int * sortIndex;
        boost::thread *sortThread;
        GAFW::FactoryStatisticsOutput *statisticManager;
        void wait_for_sort_ready();
        void sortFuncThread();
        
    public:
        SortIndex(std::string nickname,MSVectorColumnLoader<int> &antenna1, MSVectorColumnLoader<int> &antenna2, const int noOfAntennas,GAFW::FactoryStatisticsOutput *statisticManager);
        virtual ~SortIndex();
        const int * getSortIndex(); //will only return once sort is ready
        const int getNoOfRecords(); //will only return once sort is ready
        
    private:

    };
}

#endif	/* SORTINDEX_H */

