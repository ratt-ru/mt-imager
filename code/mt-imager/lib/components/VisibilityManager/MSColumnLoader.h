/* MSColumnLoader.h: MSColumnLoader template class. 
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
#ifndef __MSCOLUMNLOADER_H__
#define	__MSCOLUMNLOADER_H__
#include "ms/MeasurementSets/MeasurementSet.h"
#include "ms/MeasurementSets/MSMainColumns.h"
#include <boost/thread.hpp>

namespace mtimager
{
    template <class CasaArrayType,class TableColumn, class StorageType>
    class MSColumnLoader {
        MSColumnLoader(const MSColumnLoader& orig){};
    protected:
        CasaArrayType array;
        bool storageDeleteFlag;
        boost::condition_variable myCondition;
        boost::mutex myMutex;
        bool dataLoaded;
        void wait_for_load();
        void setAsloaded();
        StorageType * storage;
        
        boost::mutex storageCounterMutex;
        int storageCounter;
        
    public:
        MSColumnLoader();
        virtual ~MSColumnLoader();
        inline int nelements();
        virtual void loadData(const TableColumn &col); //runs directly in thread
        StorageType * getStorage(); //This function will only return storage after being loaded... else it waits
        void freeStorage();
    };
    
    template <class CasaArrayType,class TableColumn, class StorageType>
    void MSColumnLoader<CasaArrayType,TableColumn,StorageType>::wait_for_load()
    {
        boost::mutex::scoped_lock lock(this->myMutex);
        while (!dataLoaded)
        {
            myCondition.wait(lock);
        }
    }
    template <class CasaArrayType,class TableColumn, class StorageType>
    void MSColumnLoader<CasaArrayType,TableColumn,StorageType>::setAsloaded()
    {
        boost::mutex::scoped_lock lock(this->myMutex);
        this->dataLoaded=true;
        this->myCondition.notify_all();
    }
    template <class CasaArrayType,class TableColumn, class StorageType>
    MSColumnLoader<CasaArrayType,TableColumn,StorageType>::MSColumnLoader()
    {
        this->storageCounter=0;
        this->dataLoaded=false;
    }
    template <class CasaArrayType,class TableColumn, class StorageType>
    MSColumnLoader<CasaArrayType,TableColumn,StorageType>::~MSColumnLoader()
    {
        this->array.resize();
    }
    template <class CasaArrayType,class TableColumn, class StorageType>
    int MSColumnLoader<CasaArrayType,TableColumn,StorageType>::nelements()
    {
        return this->array.nelements();
    }
    template <class CasaArrayType,class TableColumn, class StorageType>
    void MSColumnLoader<CasaArrayType,TableColumn,StorageType>::loadData(const TableColumn & col)
    {
        col.getColumn(this->array,true);
        this->setAsloaded();
    }
    template <class CasaArrayType,class TableColumn, class StorageType>
    StorageType * MSColumnLoader<CasaArrayType,TableColumn,StorageType>::getStorage()
    {
        this->wait_for_load();
        boost::mutex::scoped_lock lock(this->storageCounterMutex);
        this->storageCounter++;
        if (this->storageCounter==1);
                this->storage=array.getStorage(this->storageDeleteFlag);
        return this->storage;
        
    }
    template <class CasaArrayType,class TableColumn, class StorageType>
    void MSColumnLoader<CasaArrayType,TableColumn, StorageType>::freeStorage()
    {
        
        const StorageType *store=this->storage;
        boost::mutex::scoped_lock lock(this->storageCounterMutex);
        this->storageCounter--;
        if (this->storageCounter==0)
        {        array.freeStorage(store,this->storageDeleteFlag);
                array.resize();
        }
        
    }
    template <class StoreType>
    class MSArrayColumnLoader:public MSColumnLoader<casa::Array<StoreType>,casa::ROArrayColumn<StoreType>,StoreType> {};
    template <class StoreType>
    class MSVectorColumnLoader:public MSColumnLoader<casa::Vector<StoreType>,casa::ROScalarColumn<StoreType>,StoreType> {};
    
    
}
#endif	/* MSCOLUMNLOADER_H */

