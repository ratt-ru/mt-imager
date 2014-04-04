/* mtimager.h:  A master include file, including other files related to the mt-imager  
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
#ifndef MTIMAGER_H
#define	MTIMAGER_H
#ifndef __CUDACC__
#include "GPUafw.h"
#endif
//#define USE_INTERNAL
#ifdef USE_INTERNAL
#define POLARIZATIONS 4
#endif
namespace mtimager
{
    namespace PolarizationType
    {
        enum Type
        {
            Linear,    //when X,Y
            Circular  //When R,L
        };
    }
    struct VisData
    {
        enum DataType
        {
                UVW=0,
                WEIGHT,
                FREQUENCY,
                FLAGS,
                VISIBILITY,
                TotalOutputs
        };
        static const char * names[];
    };
    
    class VisibilityManager;
    class MSVisibilityDataSetAsync;
    class MTImagerException;
    class FrequencyConverter;

};

#ifndef __CUDACC__
#include "ms/MeasurementSets/MeasurementSet.h"
#include "ms/MeasurementSets/MSMainColumns.h"
#include "measures/Measures.h"
#include "ms/MeasurementSets.h"
#include "measures/Measures/MCFrequency.h"
#include "measures/Measures/MeasTable.h"
#include "measures/TableMeasures.h"


#include "MTImagerException.h"

#include "VisibilityManager.h"
#include "MSVisibilityDataSetAsync.h"

#endif

#endif	/* MTIMAGER_H */

