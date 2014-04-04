/* EpochColumnLoader.h: Definition of the EpochColumnLoader class. 
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
#ifndef EPOCHCOLUMNLOADER_H
#define	EPOCHCOLUMNLOADER_H



#include "ms/MeasurementSets/MeasurementSet.h"
#include "ms/MeasurementSets/MSMainColumns.h"
#include <boost/thread.hpp>

namespace mtimager
{
    
    class EpochColumnLoader {
        EpochColumnLoader(const EpochColumnLoader& orig){};
    protected:
        boost::condition_variable myCondition;
        boost::mutex myMutex;
        bool dataLoaded;
        void wait_for_load();
        void setAsloaded();
        casa::MEpoch * storage;
    
        boost::mutex storageCounterMutex;
        int storageCounter;
        
    public:
        EpochColumnLoader();
        virtual ~EpochColumnLoader();
        inline int nelements();
        virtual void loadData(const casa::ROScalarMeasColumn<casa::MEpoch> &col); //runs directly in thread
        casa::MEpoch * getStorage(); 
        void freeStorage(){};
    };
};
    
    
#endif	/* EPOCHCOLUMNLOADER_H */

