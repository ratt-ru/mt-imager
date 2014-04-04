/* LogFacility.h:  Definition of the LogFacility Interface     
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
 * Author's contact details can be found on url http://www.danielmuscat.com 
 */

#ifndef __LOGFACILITY_H__
#define	__LOGFACILITY_H__
#include <string>

#include "log4cxx/logger.h"
using namespace log4cxx;
using namespace log4cxx::helpers;



typedef std::string string;
namespace GAFW 
{
class LogFacility {
private:
    Identity * thisId;
    LogFacility(const LogFacility& orig){};
public:
    enum LogType
    {
        validation=0,
        execution_submission,
        execution_performance,
        execution_memory,
        execution,
        builder,
        other
    };  //When changing this enum remember to change initialisation in LogFacility()
    string logstring[7];
    /*={
            "validation" //, "execution.submission","execution.performance","execution.memory","execution","builder","other"
        };*/
    inline LogFacility();
    virtual inline ~LogFacility();
    inline void init();
    inline void logDebug(enum LogType logtype,Identity * regarding,string message);
    inline void logDebug(enum LogType logtype,string message);
    inline void logInfo(enum LogType logtype,Identity * regarding,string message);
    inline void logInfo(enum LogType logtype,string message);
    inline void logWarn(enum LogType logtype,Identity * regarding,string message);
    inline void logWarn(enum LogType logtype,string message);
    inline void logError(enum LogType logtype,Identity * regarding,string message);
    inline void logError(enum LogType logtype,string message);
};
inline LogFacility::LogFacility()
{
    logstring[validation]="validation";
    logstring[execution_submission]="execution.submission";
    logstring[execution_performance]="execution.performance";
    logstring[execution_memory]="execution.memory";
    logstring[execution]="execution";
    logstring[builder]="builder";
    logstring[other]="other";
    thisId=NULL;
    
}
inline void LogFacility::init()
{
    thisId=dynamic_cast<Identity *>(this);
    if (thisId==NULL) throw Bug("The class which LogFacility inherits LogFacility is expected to inherit the Identity class too.");
}

inline LogFacility::~LogFacility()
{
    
}
inline void LogFacility::logDebug(enum LogType logtype,Identity * regarding,string message)
{
    LoggerPtr logger(Logger::getLogger(this->logstring[logtype]+ "." + thisId->getObjectName()));
    LOG4CXX_DEBUG(logger,string("Regarding ")+regarding->getObjectName()+": "+message);
    

}
inline void LogFacility::logDebug(enum LogType logtype,string message)
{
    LoggerPtr logger(Logger::getLogger(this->logstring[logtype]+ "." + thisId->getObjectName()));
    LOG4CXX_DEBUG(logger, message);
    

}
inline void LogFacility::logInfo(enum LogType logtype,Identity * regarding,string message)
{
    LoggerPtr logger(Logger::getLogger(this->logstring[logtype]+ "." + thisId->getObjectName()));
    LOG4CXX_INFO(logger,string("Regarding ")+regarding->getObjectName()+": "+message);
}

inline void LogFacility::logInfo(enum LogType logtype,string message)
{
    LoggerPtr logger(Logger::getLogger(this->logstring[logtype]+ "." + thisId->getObjectName()));
    LOG4CXX_INFO(logger,message);
}
inline void LogFacility::logWarn(enum LogType logtype,Identity * regarding,string message)
{
    LoggerPtr logger(Logger::getLogger(this->logstring[logtype]+ "." + thisId->getObjectName()));
    LOG4CXX_WARN(logger,string("Regarding ")+regarding->getObjectName()+": "+message);
}
inline void LogFacility::logWarn(enum LogType logtype,string message)
{
    LoggerPtr logger(Logger::getLogger(this->logstring[logtype]+ "." + thisId->getObjectName()));
    LOG4CXX_WARN(logger,message);
}
inline void LogFacility::logError(enum LogType logtype,Identity * regarding,string message)
{
    LoggerPtr logger(Logger::getLogger(this->logstring[logtype]+ "." + thisId->getObjectName()));
    LOG4CXX_ERROR(logger,string("Regarding ")+regarding->getObjectName()+": "+message);
}
inline void LogFacility::logError(enum LogType logtype,string message)
{
    LoggerPtr logger(Logger::getLogger(this->logstring[logtype]+ "." + thisId->getObjectName()));
    LOG4CXX_ERROR(logger,message);
}

} //end of namespace
#endif	/* LOGGING_H */

