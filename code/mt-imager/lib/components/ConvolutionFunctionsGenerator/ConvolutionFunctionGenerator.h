/* ConvolutionFunctionGenerator.h:  Definition of the ConvolutionFunctionGenerator base class. 
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
#ifndef __CONVOLUTIONFUNCTIONGENERATOR_H__
#define	__CONVOLUTIONFUNCTIONGENERATOR_H__
namespace mtimager
{
    class ConvolutionFunctionGenerator  {
    public:
        virtual GAFW::Result * getConvFunction()=0;
        virtual GAFW::Result * getConvFunctionPositionData()=0;
        virtual int getMaxSupport()=0;
        virtual int getSampling()=0;
        virtual float getWSquareIncrement()=0;
        virtual GAFW::Result * getConvFunctionSumData()=0;
        virtual void calculateConvFunction()=0;
    };
}

#endif	/* CONVOLUTIONFUNCTIONGENERATOR_H */

