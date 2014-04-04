/* ConfigurationManager.h:  Definition of the ConfigurationManager component and class. 
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


#ifndef CONFIGURATIONMANAGER_H
#define	CONFIGURATIONMANAGER_H
#include "ImageManager.h"
#include "WImager.h"
#include "WProjectionConvFunction.h"
#include "VisibilityManager.h"
#include "OnlyTaperConvFunction.h"
#include "PropertiesManager.h"
#include "Properties.h"
#include "casa/Quanta/Quantum.h"
#include "ImageFinalizer.h"
#include "statistics/ImagerStatistics.h"
#include <vector>
namespace mtimager
{
        class ConfigurationManager {
        private:
          ConfigurationManager(const ConfigurationManager& orig) {};
        protected:
            GAFW::Tools::CppProperties::PropertiesManager propMan;
            GAFW::Tools::CppProperties::Properties params;
            VisibilityManager *visManager;
            mtimager::statistics::ImagerStatistics *statisticsManager;
            struct {
                //int functionMinimumSupport;
                int wplanes;
                int sampling;
            } wprojection;
            std::string mode;
            //The loaded configuration
            casa::Quantity l_increment;
            casa::Quantity m_increment;
            int l_pixels;
            int m_pixels;
            std::vector<int> channels;
            std::string outputFITSFileName;
            std::vector<int> spectral_windows;
            struct 
            { 
                std::string operator_name;
                int support;
                int sampling;
                
            } taper;
            
        public:
            ConfigurationManager(int argc, char** argv);
            ImageManager::Conf getImageManagerConf();
            GAFW::Tools::CppProperties::Properties & getParams() {return this->params; } //temporary function
            ~ConfigurationManager();
            void setVisibilityManager(VisibilityManager * visMan);
            WImager::Conf getWImagerConf();
            WProjectionConvFunction::Conf getWProjectionConvFunctionConf();
            OnlyTaperConvFunction::Conf getOnlyTaperConvFunctionConf();
            ImageFinalizer::Conf getImageFinalizerConf();
            mtimager::statistics::ImagerStatistics::Conf getImagerStatisticsConf();
            void setImagerStatisticsPointer(mtimager::statistics::ImagerStatistics *statisticsManager);
            VisibilityManager::Conf getVisibilityManagerConf();
            std::string getMode();
            int getNoOfChannelsToPlot();
        private:

        };
};
#endif	/* CONFIGURATIONMANAGER_H */

