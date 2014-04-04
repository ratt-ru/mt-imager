/* ImagerTimeStatistic.cpp:  Implementation of the ImagerTimeStatistics class. 
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
#include "ImagerTimeStatistic.h"
using namespace mtimager::statistics;
using namespace GAFW;
ImagerTimeStatistic::ImagerTimeStatistic(GAFW::FactoryStatisticsOutput *statOut,GAFW::Identity * identity,std::string actionType, std::string actionDetail,long long int cycleNo)
{
    this->statOut=statOut;
    this->actionType=actionType;
    this->cycleNo=cycleNo;
    this->actionDetail=actionDetail;
    this->identity_name=identity->name;
    this->identity_objectName=identity->objectName;
    this->createData();
}
void ImagerTimeStatistic::createData()
{
    this->data=new Data;
    this->data->actionDetail=this->actionDetail;
    this->data->actionType=this->actionType;
    this->data->cycleNo=this->cycleNo;
    this->data->identity_name=this->identity_name;
    this->data->identity_objectName=this->identity_objectName;
    
}
ImagerTimeStatistic::~ImagerTimeStatistic()
{
    if (this->data!=NULL) delete this->data;
    
}
void ImagerTimeStatistic::start()
{
    if (this->data==NULL)
        this->createData();
    this->data->start_point=boost::chrono::high_resolution_clock::now();
}
void ImagerTimeStatistic::end()
{
    this->data->end_point=boost::chrono::high_resolution_clock::now();
    Statistic *stat=this->data;
    this->statOut->push_statistic(stat);
    this->data=NULL;
    
}
void ImagerTimeStatistic::changeActionData(std::string actionType, std::string actionDetail)
{
    this->actionDetail=actionDetail;
    this->actionType=actionType;
    if (this->data!=NULL)
    {
        this->data->actionDetail=actionDetail;
        this->data->actionType=actionType;
    }
}
void ImagerTimeStatistic::changeCycleNo(int cycleNo)
{
    this->cycleNo=cycleNo;
    if (this->data!=NULL)
    {
        this->data->cycleNo=cycleNo;
    }
    
}
        
