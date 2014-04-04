/* VisibilityForPSFStorageManager.h: Definition of the VisibilityForPSFStorageManager class. 
 * A special StorageManger for Visibility when PSF is requested 
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
*/

#ifndef __VISIBILITYFORPSFSTORAGEMANAGER_H__
#define	__VISIBILITYFORPSFSTORAGEMANAGER_H__

#include "SortedStorage.h"
#include "ThreadEntry.h"
namespace mtimager
{
    class VisibilityForPSFStorageManager:public GAFW::Identity,public GAFW::LogFacility ,public GAFW::FactoryAccess{
    
    public:

        VisibilityForPSFStorageManager(GAFW::Factory *factory,std::string objectName,std::string storeName, GAFW::FactoryStatisticsOutput *statisticManager,
                SortIndex &index, 
                int noOfPolarizations,
                int chunksLength);
        virtual ~VisibilityForPSFStorageManager();
        GAFW::Array * getArray(int chunkNo);
        void init_arrays();
        void arrayCreateThreadFunc();
        
     protected:
         void waitForArraysReady();
        class MyThreadEntry
        {
        protected:
            int offset;
            VisibilityForPSFStorageManager &man;
        public:
            MyThreadEntry(VisibilityForPSFStorageManager &man,int offset):offset(offset),man(man){}
            void operator()() 
            {
                man.arrayCreateThreadFunc();
            }
        };
        boost::mutex myMutex;
        boost::condition_variable myCond;
        bool arraysReady;
        GAFW::Array *mainArray;
        GAFW::Array *lastArray;
        std::string storeName;
        GAFW::FactoryStatisticsOutput *statisticManager;
        SortIndex &index; 
        int noOfPolarizations;
        int chunksLength;
        

    };
}
#endif	/* VISIBILITYFORPSFSTORAGEMANAGER_H */

