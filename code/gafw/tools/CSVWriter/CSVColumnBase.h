/* CSVColumnBase.h: Header file for the CSVColumnBase class, part of the GAFW CSVWriter Tool     
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

#ifndef CSVCOLUMNBASE_H
#define	CSVCOLUMNBASE_H
#include <string>
namespace GAFW { namespace Tools { namespace CSVWriter 
{

    class CSVColumnBase {
    protected:
        std::string name;

    public:
        CSVColumnBase(std::string name);
        CSVColumnBase(const CSVColumnBase& orig);
        virtual  std::string getTitle();
        virtual  std::string getValue()=0;
        virtual ~CSVColumnBase();
    private:

    };
} } } ;
#endif	/* CSVCOLUMNBASE_H */

