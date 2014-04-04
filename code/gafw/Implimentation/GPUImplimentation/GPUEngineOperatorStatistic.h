/* GPUEngineOperatorStatistic.h:  Definition of the Statistic class GPUEngineOperatorStatistic.h. 
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
#ifndef GPUENGINEOPERATORSTATISTIC_H
#define	GPUENGINEOPERATORSTATISTIC_H
namespace GAFW { namespace GPU {
    class GPUEngineOperatorStatistic :public GAFW::Statistic
    {
        public:
            float kernelExcecutionTime;
            std::string operatorNickname;
            std::string operatorName;
            int snapshotNo;
    };
}}
#endif	/* GPUENGINEOPERATORSTATISTIC_H */

