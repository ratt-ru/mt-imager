/* ImagerTimeStatistic.h:  Definition of the ImagerTimeStatistic imager component and class. 
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
#ifndef __IMAGERTIMESTATISTIC_H__
#define	__IMAGERTIMESTATISTIC_H__
#include "boost/chrono.hpp"
#include "GPUafw.h"
namespace mtimager { namespace statistics
{
        class ImagerTimeStatistic 
        {
        public:
            class Data: public GAFW::Statistic
            {
                public:
                long long int cycleNo;
                std::string actionType; 
                std::string actionDetail;
                std::string identity_objectName;
                std::string identity_name;
                boost::chrono::high_resolution_clock::time_point start_point;
                boost::chrono::high_resolution_clock::time_point end_point;

            };
        protected:
            GAFW::FactoryStatisticsOutput *statOut;
            bool started,ended;
            int cycleNo;
            std::string actionType; 
            std::string actionDetail;
            std::string identity_objectName;
            std::string identity_name;
            Data *data;
            void createData();
            
            
        public:
           
           ImagerTimeStatistic(GAFW::FactoryStatisticsOutput *statOut,GAFW::Identity * identity,std::string actionType="", std::string actionDetail="",long long int cycleNo=-1);
           virtual ~ImagerTimeStatistic();
           void start();
           void end();
           void changeActionData(std::string actionType, std::string actionDetail);
           void changeCycleNo(int cycleNo);
        };
}};

#endif	/* IMAGERTIMESTATISTIC_H */

