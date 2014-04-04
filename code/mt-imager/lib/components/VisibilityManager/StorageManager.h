/* StorageManager.h: StorageManager template class. 
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
#ifndef __STORAGEMANAGER_H__
#define	__STORAGEMANAGER_H__

#include "SortedStorage.h"
#include "ThreadEntry.h"

namespace mtimager
{
    template <class InputStoreType,class OutputStoreType,GAFW::GeneralImplimentation::StoreType storeType,enum  CopyStyle::copyStyleEnum copyStyle>
    class SortedStorage;
    
    template <class ColumnLoader,class InputStoreType,class OutputStoreType,GAFW::GeneralImplimentation::StoreType storeType,enum  CopyStyle::copyStyleEnum copyStyle>
    class StorageManager : public GAFW::Identity,  public GAFW::FactoryAccess 
    {
        protected:
            std::vector<boost::thread *> threads;
            InputStoreType *inputStorage;
            std::vector<int> chosenChannels;
            
            typedef SortedStorage<InputStoreType,OutputStoreType,storeType,copyStyle> MySortedStorageType; 
            std::vector <MySortedStorageType *> sortedStorageVector;
            std::string storeName;
            GAFW::FactoryStatisticsOutput *statisticManager;
            SortIndex &index; 
            int noOfColsToLoad;
            int inputRowLength;
            int chunksLength;
           int parallel_sorts;
        public:
            ColumnLoader columnLoader;
            StorageManager(GAFW::Factory *factory,std::string objectName,std::string storeName, GAFW::FactoryStatisticsOutput *statisticManager,
             SortIndex &index, 
            int noOfColsToLoad, int inputRowlength,
            int chunksLength, int parallel_sorts);
            void setNextChannelStorage(int offsetCols);
            GAFW::Array * getArray(int channelNo,int chunkNo);
            void init_sorting();
            void sortingThreadFunc();
            void sortThreadFunc(int offset);
            
            protected:
                typedef StorageManager<ColumnLoader,InputStoreType,OutputStoreType,storeType,copyStyle> MySelfType;
                class MyThreadEntry
                {
                protected:
                    int offset;
                    MySelfType &man;
                public:
                    MyThreadEntry(MySelfType &man,int offset):offset(offset),man(man){}
                    void operator()() 
                    {
                        man.sortThreadFunc(offset);
                    }
                };

            
            
    };
    template <class ColumnLoader,class InputStoreType,class OutputStoreType,GAFW::GeneralImplimentation::StoreType storeType,enum  CopyStyle::copyStyleEnum copyStyle>
    StorageManager<ColumnLoader,InputStoreType,OutputStoreType,storeType,copyStyle>::StorageManager(GAFW::Factory *factory,std::string objectName,std::string storeName, GAFW::FactoryStatisticsOutput *statisticManager,
             SortIndex &index, 
            int noOfColsToLoad, int inputRowLength,
            int chunksLength, int parallel_sorts)
                :Identity(objectName,"StorageManager"),FactoryAccess(factory),storeName(storeName),statisticManager(statisticManager),index(index),
            noOfColsToLoad(noOfColsToLoad),inputRowLength(inputRowLength),chunksLength(chunksLength),parallel_sorts(parallel_sorts)
    {
        
        FactoryAccess::init();
        this->getFactory()->registerIdentity(this);
    }
    template <class ColumnLoader,class InputStoreType,class OutputStoreType,GAFW::GeneralImplimentation::StoreType storeType,enum  CopyStyle::copyStyleEnum copyStyle>
    void StorageManager<ColumnLoader,InputStoreType,OutputStoreType,storeType,copyStyle>::setNextChannelStorage(int offsetCols)
    {
        if (offsetCols==-1) 
        {
            this->sortedStorageVector.push_back(NULL);
        }
        else
        {
            int channelNo=this->sortedStorageVector.size();
            std::stringstream channelNoString;
            channelNoString<<this->sortedStorageVector.size();
            MySortedStorageType *store=new MySortedStorageType(this->getFactory(),this->objectName+".Sorter-channel-" +channelNoString.str(),string("StorageSorter-")+this->storeName,this->statisticManager,this->index,this->noOfColsToLoad,offsetCols, this->inputRowLength,this->chunksLength);
            this->sortedStorageVector.push_back(store);
            chosenChannels.push_back(channelNo);
        }    
    }
    template <class ColumnLoader,class InputStoreType,class OutputStoreType,GAFW::GeneralImplimentation::StoreType storeType,enum  CopyStyle::copyStyleEnum copyStyle>
    GAFW::Array * StorageManager<ColumnLoader,InputStoreType,OutputStoreType,storeType,copyStyle>::getArray(int channelNo,int chunkNo)
    {
        return this->sortedStorageVector[channelNo]->getArray(chunkNo);
    }
    template <class ColumnLoader,class InputStoreType,class OutputStoreType,GAFW::GeneralImplimentation::StoreType storeType,enum  CopyStyle::copyStyleEnum copyStyle>
    void StorageManager<ColumnLoader,InputStoreType,OutputStoreType,storeType,copyStyle>::sortThreadFunc(int offset)
    {
        for (int choosenChannelNo=offset;choosenChannelNo<(int)this->chosenChannels.size();choosenChannelNo+=this->parallel_sorts)
        {
            this->sortedStorageVector[this->chosenChannels[choosenChannelNo]]->createArrays();
        }
        for (int choosenChannelNo=offset;choosenChannelNo<(int)this->chosenChannels.size();choosenChannelNo+=this->parallel_sorts)
        {
            this->sortedStorageVector[this->chosenChannels[choosenChannelNo]]->sort(this->inputStorage);
        }
    }
    template <class ColumnLoader,class InputStoreType,class OutputStoreType,GAFW::GeneralImplimentation::StoreType storeType,enum  CopyStyle::copyStyleEnum copyStyle>
    void StorageManager<ColumnLoader,InputStoreType,OutputStoreType,storeType,copyStyle>::sortingThreadFunc()
    {
        
        
        this->inputStorage=this->columnLoader.getStorage();
        for (int i=0;i<this->parallel_sorts;i++)
        {
            this->threads.push_back(new boost::thread(MyThreadEntry(*this,i)));
        }
        for (vector<boost::thread *>::iterator tr=this->threads.begin();tr<this->threads.end();tr++)
        {
            (*tr)->join();
        }
        this->columnLoader.freeStorage();
        this->inputStorage=NULL; 
    }
    template <class ColumnLoader,class InputStoreType,class OutputStoreType,GAFW::GeneralImplimentation::StoreType storeType,enum  CopyStyle::copyStyleEnum copyStyle>
    void StorageManager<ColumnLoader,InputStoreType,OutputStoreType,storeType,copyStyle>::init_sorting()
    {
        new boost::thread(ThreadEntry<MySelfType>(this,&MySelfType::sortingThreadFunc));
    }
    
            
    typedef class StorageManager<MSArrayColumnLoader<double>,double,float,GAFW::GeneralImplimentation::real_float,CopyStyle::Transpose> UVWStorageManager;
    typedef class StorageManager<MSArrayColumnLoader<std::complex<float> >,std::complex<float>,GAFW::SimpleComplex<float>,GAFW::GeneralImplimentation::real_float,CopyStyle::Normal> VisibilityStorageManager;
    typedef class StorageManager<MSArrayColumnLoader<bool>,bool,int,GAFW::GeneralImplimentation::real_int,CopyStyle::Normal> FlagsStorageManager;
    typedef class StorageManager<MSArrayColumnLoader<float>,float,float,GAFW::GeneralImplimentation::real_float,CopyStyle::Normal> WeightsStorageManager;
    //typedef class StorageManager<MSVectorColumnLoader<double>,double,double,GAFW::GeneralImplimentation::real_float,CopyStyle::Normal> TimeStorageManager;
};

    
    
    
    
    

#endif	/* STORAGEMANAGER_H */

