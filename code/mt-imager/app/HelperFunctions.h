/* 
 * File:   HelperFunctions.h
 * Author: daniel
 *
 * Created on 14 May 2012, 22:00
 */

#ifndef __HELPERFUNCTIONS_H__
#define	__HELPERFUNCTIONS_H__

#include "gafw.h"

void saveImage(GAFW::Result *src, int id, string name, int channel);
void print2D(GAFW::Result *r,int b);
void print3D(GAFW::Result *r,int b);
#endif	/* HELPERFUNCTIONS_H */

