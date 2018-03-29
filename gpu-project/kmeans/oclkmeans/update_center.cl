/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 //data[i*dim]~data[i*(dim+1)-1] --> one node data
 //so is center
 //local size = K
 //global size = K*N
__kernel void update_center(__global const float* data, __global const float* center, 
			 __global int* CdT, int N, int K, int dim, __global float* center_temp)//, __global int* points)
{
	int k;
	int i = get_global_id(0);
	//int j =1;
	int j = get_global_id(1);
	float tmp = 0.0f;
	int pts = 0;

	barrier (CLK_GLOBAL_MEM_FENCE);

	for (k=0; k<N; k=k+1){
		tmp = tmp + CdT[i*N+k] * data[k*dim+j];
		pts = pts + CdT[i*N+k];
		barrier (CLK_GLOBAL_MEM_FENCE);
	}
	tmp = tmp / pts;
	//points[i] = pts;
	//center_temp[i*dim+j] = center_temp[i*dim+j] + tmp;
	center_temp[i*dim+j] = tmp;
	
}

