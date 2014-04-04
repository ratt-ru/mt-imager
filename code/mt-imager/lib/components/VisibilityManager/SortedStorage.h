/* SortedStorage.h: SortedStorage template class. 
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
#ifndef __SORTEDSTORAGE_H__
#define	__SORTEDSTORAGE_H__
#include "SortIndex.h"
#include <boost/thread.hpp>
#include "ThreadEntry.h"
#include "MTImagerException.h"
#include "EpochColumnLoader.h"
#include "ms/MeasurementSets/MeasurementSet.h"
#include "ms/MeasurementSets/MSMainColumns.h"
#include "measures/Measures.h"
#include "ms/MeasurementSets.h"
#include "measures/Measures/MCFrequency.h"
#include "measures/Measures/MeasTable.h"
#include "measures/TableMeasures.h"
#include "gafw.h"
#include "gafw-impl.h"
#include "statistics/DetailedTimeStatistic.h"
#include <sstream>
namespace mtimager
{
    namespace CopyStyle
    {
        enum copyStyleEnum
        {
            Normal,
            UseMemCpy,
            Transpose
        };
    };
    template <class InputStoreType,class OutputStoreType,GAFW::GeneralImplimentation::StoreType storeType, enum  CopyStyle::copyStyleEnum copyStyle>
    class SortedStorage;
    
    template <class InputStoreType,class OutputStoreType,GAFW::GeneralImplimentation::StoreType storeType,enum  CopyStyle::copyStyleEnum copyStyle>
    void sortedStorageCopy(SortedStorage<InputStoreType,OutputStoreType,storeType,copyStyle> * This,int &inputRowNo, int &bufferRowNo);
    
    template<class OutputStoreType,GAFW::GeneralImplimentation::StoreType storeType, enum  CopyStyle::copyStyleEnum copyStyle>
    inline void setArray(GAFW::Array *array, int chunkLength,int noOfColsToLoad)
    {
        switch (copyStyle)
        {
            case CopyStyle::Normal:
            case CopyStyle::UseMemCpy:
                array->setDimensions(GAFW::ArrayDimensions(2,chunkLength,noOfColsToLoad));
                array->setType(GAFW::GeneralImplimentation::DataTypeManual(storeType));
                array->createStore(GAFW::GeneralImplimentation::DataTypeManual(storeType));
                break;
            case CopyStyle::Transpose:
                array->setDimensions(GAFW::ArrayDimensions(2,noOfColsToLoad,chunkLength));
                array->setType(GAFW::GeneralImplimentation::DataTypeManual(storeType));
                array->createStore(GAFW::GeneralImplimentation::DataTypeManual(storeType));
                break;
        }
        
    }
    
    template<>
    void inline setArray<GAFW::SimpleComplex<float>,GAFW::GeneralImplimentation::real_float,CopyStyle::Normal>(GAFW::Array *array, int chunkLength,int noOfColsToLoad)
    {
        array->setDimensions(GAFW::ArrayDimensions(2,chunkLength,noOfColsToLoad*2));
        array->setType(GAFW::DataType<float>());
        array->createStore(GAFW::DataType<float>());
    }
    template <class OutputStoreType ,GAFW::GeneralImplimentation::StoreType storeType>
    void inline loadBufferToArray(OutputStoreType *buffer,GAFW::Array*array,size_t size)
    {
        GAFW::PointerWrapper<OutputStoreType> p(buffer);
        array->setValues(p,size,true,0);
    }
    template <>
    void inline loadBufferToArray<GAFW::SimpleComplex<float>,GAFW::GeneralImplimentation::real_float>(GAFW::SimpleComplex<float> *buffer,GAFW::Array*array,size_t size)
    {
        GAFW::PointerWrapper<float> buffer_wrapper((float*)buffer);
       
        array->setValues(buffer_wrapper,size*2,true,0);
    }
    
    
    
    template <class InputStoreType,class OutputStoreType,GAFW::GeneralImplimentation::StoreType storeType,enum  CopyStyle::copyStyleEnum copyStyle>
    class SortedStorage : virtual public GAFW::FactoryAccess, virtual public GAFW::Identity {
        
        std::string storeName;
        GAFW::Array ** chunkArrays;
        InputStoreType *inputStore;
        OutputStoreType *buffer;
        SortIndex &index; 
        int noOfColsToLoad;
        int offsetCols;
        int chunksLength;
        int chunkUnderCalculation;
        int inputRowLength;
        int noOfRecords;
        int noOfChunks;
        boost::thread *myThread;
        boost::mutex myMutex;
        boost::condition_variable myCond;
        GAFW::FactoryStatisticsOutput *statisticManager;
        
        virtual void copy(int inputRowNo ,int bufferIndex);
        friend void sortedStorageCopy<InputStoreType,OutputStoreType,storeType,copyStyle>(SortedStorage<InputStoreType,OutputStoreType,storeType,copyStyle> * This,int &inputRowNo, int &bufferRowNo);
    public:
        SortedStorage(GAFW::Factory *factory,std::string nickname,std::string storeName,GAFW::FactoryStatisticsOutput *statisticManager,SortIndex &index, int noOfColsToLoad, int offsetCols,int inputRowlength,int chunksLength);
        //OutputStoreType * getChunkPointer(int chunkNo); //will wait if chuncks are not readay
        GAFW::Array *getArray(int chunkNo);
        void createArrays();
        void sort(InputStoreType *inputStore);
        virtual ~SortedStorage();
        
    };
        
    
    template <class InputStoreType,class OutputStoreType,GAFW::GeneralImplimentation::StoreType storeType,enum  CopyStyle::copyStyleEnum copyStyle>
    SortedStorage<InputStoreType,OutputStoreType,storeType,copyStyle>::
    SortedStorage(GAFW::Factory *factory,std::string nickname,std::string storeName, GAFW::FactoryStatisticsOutput *statisticManager,
            SortIndex &index, 
            int noOfColsToLoad, int offsetCols, int inputRowlength,
            int chunksLength)    
                :GAFW::Identity(nickname,"StorageSorter"),FactoryAccess(factory),storeName(storeName),index(index),
                statisticManager(statisticManager)
    {
           
        FactoryAccess::init();
       this->noOfColsToLoad=noOfColsToLoad;
        this->offsetCols=offsetCols;
        this->chunksLength=chunksLength;
        this->chunkUnderCalculation=0;
        //this->outputStore=NULL;
        this->inputRowLength=inputRowlength;
        this->noOfRecords=0;
        this->getFactory()->registerIdentity(this);
        
        
       // typedef SortedStorage<ColumnLoader,InputStoreType,OutputStoreType,storeType,copyStyle> mydef; 
       // this->myThread=new boost::thread(ThreadEntry<mydef>(this,&mydef::storageGenerateFuncThread));
    }
    template <class InputStoreType,class OutputStoreType, GAFW::GeneralImplimentation::StoreType storeType,enum  CopyStyle::copyStyleEnum copyStyle>
    SortedStorage<InputStoreType,OutputStoreType,storeType,copyStyle>::
    ~SortedStorage()
    {
        this->myThread->join();
        //if (this->outputStore!=NULL)
          //  delete[] this->outputStore;
    }
    template <class InputStoreType,class OutputStoreType, GAFW::GeneralImplimentation::StoreType storeType,enum  CopyStyle::copyStyleEnum copyStyle>
    void SortedStorage<InputStoreType,OutputStoreType,storeType,copyStyle>::createArrays()
    {
        this->noOfRecords=this->index.getNoOfRecords();
        this->noOfChunks=noOfRecords/this->chunksLength;
        if ((noOfChunks*this->chunksLength)!=noOfRecords) noOfChunks++;
        this->chunkArrays=new GAFW::Array *[this->noOfChunks];
        
        //An now create arrays
       {
                int chunkNo;
                scoped_timer<DetailedTimerStatistic> a(new DetailedTimerStatistic(this,"Creation of Arrays", this->storeName,this->noOfChunks-1),this->statisticManager);
                for ( chunkNo=0;chunkNo<this->noOfChunks-1;chunkNo++)
                {
                    std::stringstream arrayNickname;
                    arrayNickname<< "Chunk No"<< chunkNo;
                    this->chunkArrays[chunkNo]=this->requestMyArray(arrayNickname.str());
                    setArray<OutputStoreType,storeType,copyStyle>(this->chunkArrays[chunkNo],this->chunksLength,this->noOfColsToLoad);
                }
                {
                    //Final chunk
                    int lastChunkLength=this->noOfRecords%this->chunksLength;
                    if (lastChunkLength==0) lastChunkLength=this->chunksLength;
                    std::stringstream arrayNickname;
                    arrayNickname<< "Chunk No"<< chunkNo;
                    this->chunkArrays[this->noOfChunks-1]=this->requestMyArray(arrayNickname.str());
                    setArray<OutputStoreType,storeType,copyStyle>(this->chunkArrays[chunkNo],lastChunkLength,this->noOfColsToLoad);
                }
       }
        
        
    }
    template <class InputStoreType,class OutputStoreType, GAFW::GeneralImplimentation::StoreType storeType,enum  CopyStyle::copyStyleEnum copyStyle>
    GAFW::Array * SortedStorage<InputStoreType,OutputStoreType,storeType,copyStyle>::getArray(int chunkNo)
    {
        boost::mutex::scoped_lock lock(this->myMutex);
        while(this->chunkUnderCalculation<=chunkNo)
        {
            this->myCond.wait(lock);
        }
        return this->chunkArrays[chunkNo];
    }
    template <class InputStoreType,class OutputStoreType, GAFW::GeneralImplimentation::StoreType storeType,enum  CopyStyle::copyStyleEnum copyStyle>
    void SortedStorage<InputStoreType,OutputStoreType,storeType,copyStyle>::
         sort(InputStoreType *inputStore)
    {
        
        
        this->inputStore=inputStore;
        const int * sortIndex=this->index.getSortIndex();
        this->buffer=(OutputStoreType*)malloc(this->chunksLength*this->noOfColsToLoad*sizeof(OutputStoreType));
        
        int outRowIndex=0;
        for (int chunkNo=0;chunkNo<noOfChunks-1;chunkNo++)
        {
            scoped_timer<DetailedTimerStatistic> a(new DetailedTimerStatistic(this,"Sorting of Storage", this->storeName,chunkNo),this->statisticManager);
            for (int bufferIndex=0;bufferIndex<this->chunksLength;bufferIndex++)
            {
                this->copy(sortIndex[outRowIndex],bufferIndex);
                outRowIndex++;
            }
            loadBufferToArray<OutputStoreType,storeType>(this->buffer,this->chunkArrays[chunkNo],this->chunksLength*this->noOfColsToLoad);
            {
                boost::mutex::scoped_lock lock(this->myMutex);
                this->chunkUnderCalculation++;
                this->myCond.notify_all();
            }

        }
       {
            int bufferIndex=0;
            scoped_timer<DetailedTimerStatistic> a(new DetailedTimerStatistic(this,"Sorting of Storage", this->storeName,this->noOfChunks-1),this->statisticManager);
            for (;outRowIndex<noOfRecords;outRowIndex++)
            {
                 this->copy(sortIndex[outRowIndex],bufferIndex);
                 bufferIndex++;
            }
            loadBufferToArray<OutputStoreType,storeType>(this->buffer,this->chunkArrays[this->noOfChunks-1],bufferIndex*this->noOfColsToLoad);
            {
                    boost::mutex::scoped_lock lock(this->myMutex);
                    this->chunkUnderCalculation++;
                    this->myCond.notify_all();
            }
       }
        free(this->buffer);
        this->buffer=NULL;
    }
    /*
    template <class InputStoreType,class OutputStoreType,GAFW::GeneralImplimentation::StoreType storeType,enum  CopyStyle::copyStyleEnum copyStyle>
    inline void sortedStorageCopy(SortedStorage<InputStoreType,OutputStoreType,storeType,copyStyle> * This,GAFW::Array * array,int inputRowNo,int arrayRowIndex) 
    {
        int inputBegin;
    
        switch (copyStyle)
        {
            case CopyStyle::Normal:
                inputBegin=inputRowNo*This->inputRowLength+This->offsetCols;
                for (int i=0;i<This->noOfColsToLoad;i++)
                {
                    array->setValue((OutputStoreType)This->inputStore[inputBegin+i],2,i,arrayRowIndex);
                    
                }
                break;
            case CopyStyle::UseMemCpy:
                inputBegin=inputRowNo*This->inputRowLength+This->offsetCols;
                //outputBegin=outputRowNo*This->noOfColsToLoad;
                //memcpy(This->outputStore+outputBegin,This->inputStore+inputBegin,This->noOfColsToLoad*sizeof(InputStoreType));
                array->setValues((OutputStoreType*)This->inputStore+inputBegin,This->noOfColsToLoad,true,2,0,arrayRowIndex);
                break;
            case CopyStyle::Transpose:
                
                inputBegin=inputRowNo*This->inputRowLength+This->offsetCols;
                for (int i=0;i<This->noOfColsToLoad;i++)
                {       
                     array->setValue((OutputStoreType)This->inputStore[inputBegin+i],2,arrayRowIndex,i);
                }
                break;
                
        }
    }
    template <>
    inline void sortedStorageCopy<std::complex<float>,GAFW::SimpleComplex<float>,GAFW::GeneralImplimentation::real_float,CopyStyle::Normal>
        (SortedStorage<std::complex<float>,GAFW::SimpleComplex<float>,GAFW::GeneralImplimentation::real_float,CopyStyle::Normal> * This,GAFW::Array * array,int inputRowNo,int arrayRowIndex)
    {
        int inputBegin;
        int outputBegin;
        inputBegin=inputRowNo*This->inputRowLength+This->offsetCols;
        outputBegin=0;
        for (int i=0;i<This->noOfColsToLoad;i++)
        {
                array->setValue(This->inputStore[inputBegin+i].real(),2,outputBegin,arrayRowIndex);
                outputBegin++;
                array->setValue(This->inputStore[inputBegin+i].imag(),2,outputBegin,arrayRowIndex);
                outputBegin++;
        }
    }
    */
    template <class InputStoreType,class OutputStoreType,GAFW::GeneralImplimentation::StoreType storeType,enum  CopyStyle::copyStyleEnum copyStyle>
    inline void sortedStorageCopy(SortedStorage<InputStoreType,OutputStoreType,storeType,copyStyle> * This,int &inputRowNo, int &bufferRowNo) 
    {
        int inputBegin;
        int outputBegin;
        switch (copyStyle)
        {
            case CopyStyle::Normal:
                inputBegin=inputRowNo*This->inputRowLength+This->offsetCols;
                outputBegin=bufferRowNo*This->noOfColsToLoad;
                for (int i=0;i<This->noOfColsToLoad;i++)
                {
                    This->buffer[outputBegin+i]=(OutputStoreType)This->inputStore[inputBegin+i];
                }
                break;
            case CopyStyle::UseMemCpy: //Use this mode only if InputStoreType is the same as OutputStoreType
                inputBegin=inputRowNo*This->inputRowLength+This->offsetCols;
                outputBegin=bufferRowNo*This->noOfColsToLoad;
                memcpy(This->buffer+outputBegin,This->inputStore+inputBegin,This->noOfColsToLoad*sizeof(InputStoreType));
            break;
            case CopyStyle::Transpose:
                int myChunkLength=This->chunksLength;
                if (This->chunkUnderCalculation==This->noOfChunks-1) 
                {
                    myChunkLength=This->noOfRecords%myChunkLength;
                    if (myChunkLength==0) myChunkLength=This->chunksLength;
                }
                inputBegin=inputRowNo*This->inputRowLength+This->offsetCols;
                for (int i=0;i<This->noOfColsToLoad;i++)
                {       
                        This->buffer[i*myChunkLength+bufferRowNo]=(OutputStoreType)This->inputStore[inputBegin+i];
                }
                break;
                
        }
    }
    template <>
    inline void sortedStorageCopy< std::complex<float>,GAFW::SimpleComplex<float>,GAFW::GeneralImplimentation::real_float,CopyStyle::Normal >
        (SortedStorage<std::complex<float>,
            GAFW::SimpleComplex<float>,GAFW::GeneralImplimentation::real_float,CopyStyle::Normal> * This,int &inputRowNo,int &bufferRowNo)
    {
        int inputBegin;
        int outputBegin;
        inputBegin=inputRowNo*This->inputRowLength+This->offsetCols;
        outputBegin=bufferRowNo*This->noOfColsToLoad;
        for (int i=0;i<This->noOfColsToLoad;i++)
        {
                This->buffer[outputBegin+i].real=This->inputStore[inputBegin+i].real();
                This->buffer[outputBegin+i].imag=This->inputStore[inputBegin+i].imag();

        }
    }
    
    template <class InputStoreType,class OutputStoreType,GAFW::GeneralImplimentation::StoreType storeType, enum  CopyStyle::copyStyleEnum copyStyle >
    void SortedStorage<InputStoreType,OutputStoreType,storeType,copyStyle>::
        copy(int inputRowNo ,int bufferIndex)
    {
        sortedStorageCopy<InputStoreType,OutputStoreType,storeType,copyStyle>(this,inputRowNo,bufferIndex);
    }
    
        
           
}

#endif	/* SORTEDSTORAGE_H */

