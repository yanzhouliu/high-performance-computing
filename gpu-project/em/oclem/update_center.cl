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


//float Matrix_det( float* C){
//	return C[0]*C[3] - C[1]*C[2];
//}

float Matrix_det(float c0, float c1, float c2, float c3, int iGiD){
	float x=c0*c3-c1*c2;
	return x;
	if(iGiD == 0){
		return (c0*c3-c1*c2);
	}else{
		return -1;
	}
}


void Matrix_inv( float* covx, float* covx_inv){
	float det = covx[0]*covx[3] - covx[1]*covx[2];
	covx_inv[0] = covx[3]/det;
	covx_inv[1] = -covx[1]/det;
	covx_inv[2] = -covx[2]/det;
	covx_inv[3] = covx[0]/det;

}



__kernel void update_center(__global const float* data, __global float* covx,
							__global float* covx_inv, int N, int dim )
{
	int p;
	int iLiD = get_local_id(0);
	int iGiD = get_global_id(0);
	float mu[2];
	float temp;
	float c[4];
	mu[iLiD] = 0;
	for(p = 0; p<N; p++){
		mu[iLiD] = mu[iLiD] + data[p*dim+iLiD];
	}
	mu[iLiD] = mu[iLiD]/N;

	covx[iGiD] = 0;
	barrier (CLK_LOCAL_MEM_FENCE);

	if(iGiD==0){  
		for(p = 0; p<N; p=p+1){
			covx[iGiD] = covx[iGiD] + (data[p*dim+iGiD]-mu[iGiD])*(data[p*dim+iGiD]-mu[iGiD]);
		}
	}
barrier (CLK_LOCAL_MEM_FENCE);
	if(iGiD==1){
		for(p = 0; p<N; p=p+1){
			covx[iGiD] = covx[iGiD] + (data[p*dim+iGiD]-mu[iGiD])*(data[p*dim+iGiD-1]-mu[iGiD-1]);
			//covx[iGiD] = 0;
		}
	}
barrier (CLK_LOCAL_MEM_FENCE);
	if(iGiD==2) {
		for(p = 0; p<N; p=p+1){
			covx[iGiD] = covx[iGiD] + (data[p*dim+iGiD-1]-mu[iGiD-1])*(data[p*dim+iGiD-2]-mu[iGiD-2]);
			//covx[iGiD] = 1.0;
		}
	}
barrier (CLK_LOCAL_MEM_FENCE);
	if(iGiD==3){
		for(p = 0; p<N; p=p+1){
			covx[iGiD] = covx[iGiD] + (data[p*dim+iGiD-2]-mu[iGiD-2])*(data[p*dim+iGiD-2]-mu[iGiD-2]);
		}
	}
	barrier (CLK_GLOBAL_MEM_FENCE);
	//temp = covx[iGiD];
	covx[iGiD] = covx[iGiD]/(N-1);
	//barrier (CLK_GLOBAL_MEM_FENCE);
barrier (CLK_LOCAL_MEM_FENCE);
	//Matrix_inv( *covx, *covx_inv);
	if(iGiD == 0){
		c[0] = covx[0];
		c[1] = covx[1];
		c[2] = covx[2];
		c[3] = covx[3];
		//temp = Matrix_det(covx[0],covx[1],covx[2],covx[3]);
		temp = Matrix_det(c[0],c[1],c[2],c[3],iGiD);
		//temp = c[0]*c[3]-c[1]*c[2];
		covx[0] = temp;
	}
barrier (CLK_LOCAL_MEM_FENCE);
	//covx[iLiD] = mu[iLiD];

	//for (k=0; k<N; k=k+1){
	//	tmp = tmp + CdT[i*N+k] * data[k*dim+iGiD];
	//	pts = pts + CdT[i*N+k];
	//	barrier (CLK_GLOBAL_MEM_FENCE);
	//}
	//tmp = tmp / pts;
	//points[i] = pts;
	//center_temp[i*dim+iGiD] = center_temp[i*dim+iGiD] + tmp;
	//center_temp[i*dim+iGiD] = tmp;
	
}

