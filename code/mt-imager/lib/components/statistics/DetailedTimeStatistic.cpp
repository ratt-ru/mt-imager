/* DetailedTimeStatistics.cpp:  Implementation of the DetailedTimeStatistic class. 
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


#include "DetailedTimeStatistic.h"
using namespace mtimager;

DetailedTimerStatistic::DetailedTimerStatistic(GAFW::Identity * identity,std::string actionType, std::string actionDetail,long long int cycleNo)
        :cycleNo(cycleNo),actionType(actionType),actionDetail(actionDetail),identity_objectName(identity->objectName),identity_name(identity->name) 
{
    
}
DetailedTimerStatistic::DetailedTimerStatistic(std::string actionType, std::string actionDetail,long long int cycleNo)
        :cycleNo(cycleNo),actionType(actionType),actionDetail(actionDetail),identity_objectName(""),identity_name("") 
{
    
}
DetailedTimerStatistic::~DetailedTimerStatistic()
{
    
}
void DetailedTimerStatistic::start()
{
    this->start_point=boost::chrono::high_resolution_clock::now();
}
void DetailedTimerStatistic::stop()
{
    this->end_point=boost::chrono::high_resolution_clock::now();
}


