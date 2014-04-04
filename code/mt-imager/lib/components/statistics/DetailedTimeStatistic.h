/* DetailedTimeStatistics.h:  Definition of the DetailedTimeStatistic class. 
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

#ifndef __DETAILEDTIMESTATISTIC_H__
#define	__DETAILEDTIMESTATISTIC_H__
#include "gafw.h"
#include <boost/chrono.hpp>
#include "scoped_timer.h"
namespace mtimager
{
    class DetailedTimerStatistic : public GAFW::Statistic 
    {
        private:
            DetailedTimerStatistic(const DetailedTimerStatistic& orig):cycleNo(-1){};
        public:
            const long long int cycleNo;
            const std::string actionType; 
            const std::string actionDetail;
            const std::string identity_objectName;
            const std::string identity_name;
            boost::chrono::high_resolution_clock::time_point start_point;
            boost::chrono::high_resolution_clock::time_point end_point;
            DetailedTimerStatistic(GAFW::Identity * identity,std::string actionType, std::string actionDetail,long long int cycleNo);
            DetailedTimerStatistic(std::string actionType, std::string actionDetail,long long int cycleNo);
            virtual ~DetailedTimerStatistic();
            void start();
            void stop();

    };
    typedef scoped_timer<DetailedTimerStatistic> scoped_detailed_timer;
};
#endif	/* DETAILEDTIMESTATISTIC_H */

