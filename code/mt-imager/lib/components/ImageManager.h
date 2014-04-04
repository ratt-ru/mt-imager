/* ImageManager.h: Definition  of the ImageManager component and class. 
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
#ifndef __IMAGEMANAGER_H__
#define	__IMAGEMANAGER_H__
#include "casa/Quanta.h"
#include "measures/Measures.h"
#include "measures/Measures/MDirection.h"
#include "measures/Measures/MFrequency.h"
#include "measures/Measures/MEpoch.h"
#include "coordinates/Coordinates.h"
#include <coordinates/Coordinates/CoordinateSystem.h>
#include <coordinates/Coordinates/DirectionCoordinate.h>
#include <coordinates/Coordinates/SpectralCoordinate.h>
#include <coordinates/Coordinates/StokesCoordinate.h>
#include <coordinates/Coordinates/Projection.h>
#include <coordinates/Coordinates/ObsInfo.h>
#include <measures/Measures/Stokes.h>
#include <casa/Quanta/UnitMap.h>
#include <casa/Quanta/UnitVal.h>
#include <casa/Quanta/MVAngle.h>
#include <casa/Quanta/MVAngle.h>
#include "gafw.h"
#include "fitsio.h"
#include <boost/thread.hpp>
#include "statistics/ImagerStatistics.h"


namespace mtimager
{


    class ImageManager {
    public:
        class Conf
        {
            public:
                bool initialized;
                double l_increment;  //pixel length 
                int lpixels; //image length in pixels
                
                double m_increment;  //pixel width
                int mpixels; //image width in pixels
                casa::MDirection phaseCenter;
                std::vector<casa::MFrequency> freqs;
                casa::MFrequency::Types obsFreqRef;
                // std::vector<int> spectralwindowids; //Imight use 
                casa::Quantity restFreq;
                //casa::MDirection TrackDir; //I might use this
                std::string telescope;
                std::vector<int> stokes; 
                casa::MEpoch obsEpoch;
                std::string observer;
                std::string outputFITSFileName;
                statistics::ImagerStatistics * stat;
                 Conf(){ initialized=false; };
                ~Conf() {};
        };
        float *tempImage;
    private:
        //Deny copy
        ImageManager(const ImageManager& orig) {};
    protected:
        std::vector<GAFW::Result*> imageRes;
        Conf conf;
        casa::CoordinateSystem coord;
        fitsfile * outputFITSImage;
        boost::mutex myMutex;
        boost::condition_variable myCond;
        int readyChannelNo;
        boost::thread *myThread;
        void createFITS();
        void saveChannel(int channelNo);
    public:
        //void saveImage(GAFW::Result * imageRes,int snapshot);
        //inline void saveImage(GAFW::Result * imageRes);
        void nextChannelReady();
        void threadFunc();
        ImageManager(const Conf conf,std::vector<GAFW::Result*> imageRes);
        ~ImageManager();

    };
    
};
#endif	/* IMAGEMANAGER_H */

