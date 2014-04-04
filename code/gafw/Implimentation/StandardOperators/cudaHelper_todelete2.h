/* 
 * File:   cudaHelper.h
 * Author: daniel
 *
 * Created on 21 November 2012, 19:04
 */

#ifndef __CUDAHELPER_H__
#define	__CUDAHELPER_H__
#include <cuda_runtime.h>
//This is a special file containing special cuda functions that might be of general help when developing

#define complex_to_cuComplex(value) make_float2(value.real(),value.imag())

#define loc_2_pos3D(loc,dim) (loc.z*dim.x*dim.y+loc.y*dim.x+loc.x)

__device__ __inline__ int  OriginToCentre1D(int& loc,int& dim)
{
    int newloc;
    if (dim%2)  //odd number
    {
        if (loc<=dim/2) newloc=loc+dim/2;
        else newloc=loc-(dim/2)-1;
    }
    else    //even dimension
    {
       if (loc<(dim/2)) newloc=loc+(dim/2);
       else newloc=loc-(dim/2); 
    }
    return newloc;
}
__device__ __inline__ int OriginToCorner1D(int & loc, int &dim)
{
    int newloc;
    if (dim%2)  //odd number
    {
        if (loc<dim/2) newloc=loc+dim/2+1;
        else newloc=loc-(dim/2);
    }
    else    //even dimension
    {
       if (loc<(dim/2)) newloc=loc+(dim/2);
       else newloc=loc-(dim/2); 
    }
    return newloc;
}

//atomicAdd for double is not supported yet by the CUDA API. This is an implientation as suggested in the guide
__device__ __inline__ double atomicAddDouble(double*,double);

__device__ __inline__ double atomicAddDouble(double* address, double val)
{
        unsigned long long int* address_as_ull =(unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val +
                __longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
}

template <class A> 
__device__ __inline__ A zero()
{
    return 0;
}
template <>
__device__ __inline__ cuComplex zero()
{
    return make_float2(0.0f,0.0f);
}
template <>
__device__ __inline__ cuDoubleComplex zero()
{
    return make_double2(0.0,0.0);
}





#endif	/* CUDAHELPER_H */

