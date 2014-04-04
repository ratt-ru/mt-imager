/* CSVColumn.h: Header file for the generic CSVColumn class, part of the GAFW CSVWriter Tool     
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

#ifndef CSVCOLUMN_H
#define	CSVCOLUMN_H
#include "CSVColumnBase.h"
#include <sstream>
#include <iomanip>
namespace GAFW { namespace Tools { namespace CSVWriter
{
    template <class T>
    class CSVColumn: public CSVColumnBase {
    protected:
        int precision;
        int width;
        T value;
        bool isFixed;
        bool isScientific;
        bool rightJustify;
        char fillChar;
    public:

        CSVColumn(std::string name=""):CSVColumnBase(name)
        {
            this->precision=3;
            this->width=0;
                
            this->isFixed=false;
            this->isScientific=false;
            this->rightJustify=false;
            this->fillChar=' ';
        }

        CSVColumn(const CSVColumn& orig):CSVColumnBase(orig)
        {
            this->precision=orig.precision;
            this->width=orig.width;
                
            this->isFixed=orig.isFixed;
            this->isScientific=orig.isScientific;
            this->rightJustify=orig.rightJustify;
            this->fillChar=orig.fillChar;
        };
        virtual void setValue(const T &value)
        {
            this->value=value;
        }
        virtual void setPrecision(int pres)
        {
            this->precision=pres;
        }
        virtual void setWidth(int width)
        {
            this->width=width;
        }
        virtual void setRightJustify()
        {
            this->rightJustify=true;
        }
        virtual void setFixed(bool isFixed=true)
        {
            this->isFixed=true;
        }

        virtual void setScientific(bool isScientific=true)
        {
            this->isScientific=isScientific;
        }
        virtual void setFill(char c)
        {
            this->fillChar=c;
        }
        virtual ~CSVColumn() {}
        void setStreamFormatting(std::stringstream &s)
        {
            s <<std::setw(this->width)<<
                std::setprecision(precision) <<
                    std::setfill(fillChar);;
            if (isFixed) s<<std::fixed; 
                    
            if (isScientific) s<<std::scientific;
             if (this->rightJustify) s<< std::right;
        }
        virtual std::string getValue()
        {
            std::stringstream s;
            setStreamFormatting(s);
            s<<value;
            return std::string(s.str());
        }
    private:

    };
}}};

#endif	/* CSVCOLUMN_H */

