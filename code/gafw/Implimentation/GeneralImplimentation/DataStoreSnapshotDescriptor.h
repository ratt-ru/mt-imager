/* Array.h:  Header file for the definition of the DataStoreSnapshotDescriptor.
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

#ifndef __GEN_DATASTORESNAPHOT_H__
#define	__GEN_DATASTORESNAPHOT_H__
namespace GAFW { namespace GeneralImplimentation {
    

    class DataStoreSnapshotDescriptor 
    {
    public:

        void *pointer;  //beginning of snapshot memory pointer
        ArrayDimensions dim; //the array dimensions of the snapshot
        DataStore *parent; //The matrix to whom the snapshot belongs
        int size;
        StoreType type;
        int snapshot_id;
        int otherId;   //An id to be used by choice.. In GPUFramework this is used as to indicate that a same snapshot has been performed with another snaphot id 
        DataStoreSnapshotDescriptor& operator=(DataStoreSnapshotDescriptor orig);

    };

} }
#endif	/* DATASTORESNAPHOT_H */

