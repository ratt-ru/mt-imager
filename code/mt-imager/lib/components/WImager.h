/* WImager.h: Definition  of the WImager component, GAFW module, and class. 
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

#ifndef __WIMAGER_H__
#define	__WIMAGER_H__
#include "gafw.h"
#include "ConvolutionFunctionGenerator.h"
#include <vector>
namespace mtimager
{
    class WImager: public GAFW::Module {
    public:
        class Conf
        {
        public:
            int u_pixels;
            int v_pixels;
            double u_increment;
            double v_increment;
            int no_of_polarizations;
            GAFW::FactoryStatisticsOutput *statisticsManager;
        };
        class WImagerStatistic: public GAFW::Statistic
        {
        public:
            int noOfPolarizations;
            int snapshotNo;
            int noOfRecords;
            int compressedRecords;
            int griddedRecords;
            long int compressedGridPoints;
            long int griddedGridPoints;
        };
    protected:
        std::vector<std::pair<int,int> > snapshots; //first sanpshot no second noOfRecords
        
        Conf conf;
        //ImageManagerOld &imagedef;
        ConvolutionFunctionGenerator * convGenerator;
       // int no_of_polarizations;
        void checkConf();
        
    public:

       WImager(GAFW::Factory *factory,std::string nickname, ConvolutionFunctionGenerator * generator,Conf conf);
       ~WImager();
       virtual void reset();
       virtual void calculate();
       virtual void setInput(int inputNo, GAFW::Result *res);
       virtual GAFW::Result * getOutput(int outputNo); 
       virtual void resultRead(GAFW::ProxyResult *,int snapshot_no);
       virtual void generateStatistics(); 
       //virtual int getNoOfPointsGridded();
       //virtual int getNoOfPointsCompressed();
    };
}
#endif	/* WIMAGER_H */

