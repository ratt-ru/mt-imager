/* EpochColumnLoader.cpp: Implementation the EpochColumnLoader class. 
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

#include "EpochColumnLoader.h"
using namespace mtimager;
void EpochColumnLoader::wait_for_load()
{
    boost::mutex::scoped_lock lock(this->myMutex);
    while (!dataLoaded)
    {
        this->myCondition.wait(lock);
    }
}
void EpochColumnLoader::setAsloaded()
{
    boost::mutex::scoped_lock lock(this->myMutex);
    this->dataLoaded=true;
    this->myCondition.notify_all();
}
EpochColumnLoader::EpochColumnLoader()
{
    this->dataLoaded=false;
    this->storage=NULL;
}
EpochColumnLoader::~EpochColumnLoader()
{
    if (this->storage!=NULL) delete[] this->storage;
}
void EpochColumnLoader::loadData(const casa::ROScalarMeasColumn<casa::MEpoch> & col)
{
    int noOfRows=col.table().nrow();
    this->storage=new casa::MEpoch[noOfRows];
    for (int i=0;i<noOfRows;i++)
    {
        col.get(i,this->storage[i]);
    }
    this->setAsloaded();
}
casa::MEpoch * EpochColumnLoader::getStorage()
{
    this->wait_for_load();
    return this->storage;

}
