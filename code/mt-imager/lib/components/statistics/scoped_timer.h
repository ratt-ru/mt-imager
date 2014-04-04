/* scoped_timer.h:  scoped_timer template wrapper.
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
#ifndef SCOPED_TIMER_H
#define	SCOPED_TIMER_H
#include "gafw.h"
namespace mtimager
{
    template <class TimerStatistic>
    class scoped_timer 
    {
        private:
            scoped_timer(const scoped_timer& orig){};
        protected:
            TimerStatistic *stat;
            GAFW::FactoryStatisticsOutput *statisticManager;
        public:
            scoped_timer(TimerStatistic *stat,GAFW::FactoryStatisticsOutput *statisticManager=NULL);
            virtual ~scoped_timer();
    };
    template <class TimerStatistic>
    scoped_timer<TimerStatistic>::scoped_timer(TimerStatistic *stat,GAFW::FactoryStatisticsOutput *statisticManager)
    {
        this->stat=stat;
        this->statisticManager=statisticManager;
        this->stat->start();
    }
    template <class TimerStatistic>
    scoped_timer<TimerStatistic>::~scoped_timer()
    {
        this->stat->stop();
        if (this->statisticManager!=NULL)
            this->statisticManager->push_statistic(this->stat);
    }
    
};
#endif	/* SCOPED_TIMER_H */

