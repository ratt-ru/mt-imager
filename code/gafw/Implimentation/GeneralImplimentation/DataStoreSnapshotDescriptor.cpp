/* DataStoreSnapshotDescriptor.cpp:  Code for DataStoreSnapshotDescriptor
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

#include "gafw-impl.h"
using namespace GAFW::GeneralImplimentation;
DataStoreSnapshotDescriptor& DataStoreSnapshotDescriptor::operator=(DataStoreSnapshotDescriptor orig)
{
    this->pointer=orig.pointer;  //beginning of snapshot memory pointer
    this->dim=orig.dim; //the array dimensions of the snapshot
    this->parent=orig.parent; //The matrix to whom the snapshot belongs
    this->size=orig.size;
    this->type=orig.type;
    this->snapshot_id=orig.snapshot_id;
    this->otherId=orig.otherId; 
}