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
//
//float Matrix_det(float c0, float c1, float c2, float c3){
//	return c0*c3-c1*c2;
//}
//
//
//void Matrix_inv( float* covx, float* covx_inv){
//	float det = covx[0]*covx[3] - covx[1]*covx[2];
//	covx_inv[0] = covx[3]/det;
//	covx_inv[1] = -covx[1]/det;
//	covx_inv[2] = -covx[2]/det;
//	covx_inv[3] = covx[0]/det;
//
//}



__kernel void update_center(__global const float* data, __global float* covx,// __global float* covx_inv,
							__global float* meanDiff, __global float* pdf, __global float* mu, __global float *phi,
							__global float *pdf_w, __global float *W, int N, int dim, int K )
{
	int p, i, Dj;
	int iLiD = get_local_id(0);
	int iGiD = get_global_id(0);
	float mu_temp[2];
	float covx_inv[4];
	float det;
	
	/*  calculate the mean, equal to matlab sum(data,1)    */
	float mean[2];
	
	Dj = dim*dim;

	barrier (CLK_GLOBAL_MEM_FENCE);

	
	for(i = 0; i < K; i++){
		//if(iGiD == 0){
		mu_temp[0] = mu[i*2];
		mu_temp[1] = mu[i*2+1];
		meanDiff[iGiD] = 0.0f;
		//}
		meanDiff[iGiD] = data[iGiD] - mu_temp[iGiD%2];
		barrier (CLK_GLOBAL_MEM_FENCE);

		//mean[iLiD] = 0;
		//for(p = 0; p<N; p++){
		//	mean[iLiD] = mean[iLiD] + data[p*dim+iLiD];
		//}
		//mean[iLiD] = mean[iLiD]/N;	
		//covx[iGiD+ Dj*i] = 0.0f;
		//barrier (CLK_LOCAL_MEM_FENCE);

		//if(iGiD==0)	{for(p = 0; p<N; p=p+1)	covx[iGiD + Dj*i] = covx[iGiD + Dj*i] + (data[p*dim+iGiD]-mean[iGiD])*(data[p*dim+iGiD]-mean[iGiD]);}
	
		//if(iGiD==1) {for(p = 0; p<N; p=p+1)	covx[iGiD + Dj*i] = covx[iGiD + Dj*i] + (data[p*dim+iGiD]-mean[iGiD])*(data[p*dim+iGiD-1]-mean[iGiD-1]);}

		//if(iGiD==2) {for(p = 0; p<N; p=p+1)	covx[iGiD + Dj*i] = covx[iGiD + Dj*i] + (data[p*dim+iGiD-1]-mean[iGiD-1])*(data[p*dim+iGiD-2]-mean[iGiD-2]);}

		//if(iGiD==3) {for(p = 0; p<N; p=p+1) covx[iGiD + Dj*i] = covx[iGiD + Dj*i] + (data[p*dim+iGiD-2]-mean[iGiD-2])*(data[p*dim+iGiD-2]-mean[iGiD-2]);}

		//barrier (CLK_GLOBAL_MEM_FENCE);
		//covx[iGiD+ Dj*i] = covx[iGiD+ Dj*i]/(N-1);
		
		//if(iGiD == 0){
			det = covx[0+ Dj*i]*covx[3+ Dj*i] - covx[1+ Dj*i]*covx[2+ Dj*i];
			covx_inv[0] = covx[3+ Dj*i]/det;
			covx_inv[1] = 0 - covx[1+ Dj*i]/det;
			covx_inv[2] = 0 - covx[2+ Dj*i]/det;
			covx_inv[3] = covx[0+ Dj*i]/det;
		//}
		barrier (CLK_GLOBAL_MEM_FENCE);
		if(iGiD % 2 == 0){
			pdf[(iGiD/2)*K + i] = (1/sqrt( pow(2*M_PI,dim) * det)) * exp( -0.5* ((meanDiff[iGiD]*covx_inv[0]+meanDiff[iGiD+1]*covx_inv[2])*meanDiff[iGiD] + (meanDiff[iGiD]*covx_inv[1]+meanDiff[iGiD+1]*covx_inv[3])*meanDiff[iGiD+1]));
			barrier (CLK_GLOBAL_MEM_FENCE);	
			pdf_w[(iGiD/2)*K + i] = pdf[(iGiD/2)*K + i] * phi[i];
		}
	}

	barrier (CLK_GLOBAL_MEM_FENCE);
	
	for(i=0; i < K; i++){
		if (iGiD % 2 == 0) 
			W[(iGiD/2)*K + i] = pdf_w[(iGiD/2)*K + i] / (pdf_w[(iGiD/2)*K + 0]+ pdf_w[(iGiD/2)*K + 1] + pdf_w[(iGiD/2)*K + 2]);
	}
	barrier (CLK_GLOBAL_MEM_FENCE);

}

