/* VisibilityForPSFStorageManager.h: Implementation of the VisibilityForPSFStorageManager class. 
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

#include "VisibilityForPSFStorageManager.h"
using namespace GAFW;
using namespace mtimager;
VisibilityForPSFStorageManager::VisibilityForPSFStorageManager(GAFW::Factory *factory,std::string objectName,std::string storeName, GAFW::FactoryStatisticsOutput *statisticManager,
                SortIndex &index, 
                int noOfPolarizations,
                int chunksLength): Identity(objectName,"StorageManager"),FactoryAccess(factory),storeName(storeName),statisticManager(statisticManager),index(index),
                noOfPolarizations(noOfPolarizations),chunksLength(chunksLength)
    {
        
        FactoryAccess::init();
        this->getFactory()->registerIdentity(this);
        this->arraysReady=false;
    }



VisibilityForPSFStorageManager::~VisibilityForPSFStorageManager() 
{

}

GAFW::Array * VisibilityForPSFStorageManager::getArray(int chunkNo)
{
    this->waitForArraysReady();
    if ((this->index.getNoOfRecords()-chunkNo*this->chunksLength)<this->chunksLength)
        return this->lastArray;
    else
        return this->mainArray;
          
}
void VisibilityForPSFStorageManager::init_arrays()
{
        new boost::thread(ThreadEntry<VisibilityForPSFStorageManager>(this,&VisibilityForPSFStorageManager::arrayCreateThreadFunc));
}
void VisibilityForPSFStorageManager::waitForArraysReady()
{
    boost::mutex::scoped_lock lock(this->myMutex);
    while(!this->arraysReady)
    {
        this->myCond.wait(lock);
    }
}
void VisibilityForPSFStorageManager::arrayCreateThreadFunc()
{
    float _one=1;
    float _zero=0;
    ValueWrapper<float> one(_one);
    ValueWrapper<float> zero(_zero);
    
    this->mainArray=this->requestMyArray("mainArray",ArrayDimensions(2,this->chunksLength,this->noOfPolarizations*2),DataType<float >());
    this->mainArray->createStore();
    
    
    for (int i=0;i<this->noOfPolarizations;i++)
        for (int j=0; j<this->chunksLength;j++)
        {    
            this->mainArray->setValue(one,2,i*2,j);
            this->mainArray->setValue(zero,2,i*2+1,j);
        }   
    
    
    if (this->index.getNoOfRecords()%this->chunksLength!=0)
    {
        this->lastArray=this->requestMyArray("lastArray",ArrayDimensions(2,this->index.getNoOfRecords()%this->chunksLength,this->noOfPolarizations*2),DataType<float>());
        this->lastArray->createStore();
        for (int i=0;i<this->noOfPolarizations;i++)
            for (int j=0; j<this->index.getNoOfRecords()%this->chunksLength;j++)
            {    
                this->lastArray->setValue(one,2,i*2,j);
                this->lastArray->setValue(zero,2,i*2+1,j);
            }
        
    }
    
    boost::mutex::scoped_lock lock(this->myMutex);
    this->arraysReady=true;
    this->myCond.notify_all();
    
    
}  
