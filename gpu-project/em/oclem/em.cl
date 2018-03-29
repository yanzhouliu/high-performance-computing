
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
 
//local:2
//global:M*N
__kernel void expectation(__global const double* data, __global double* covx,
							__global double* meanDiff, __global double* pdf, __global double* mu, __global double *phi,
							__global double *pdf_w, __global double *W, int N, int dim, int K )
{
	int p, i, Dj;
	int iLiD = get_local_id(0);
	int iGiD = get_global_id(0);
	double mu_temp[2];
	double covx_inv[4];
	double det;
	
	/*  calculate the mean, equal to matlab sum(data,1)    */
	double mean[2];
	
	Dj = dim*dim;

	//barrier (CLK_GLOBAL_MEM_FENCE);

	
	for(i = 0; i < K; i++){

		mu_temp[0] = mu[i*2];
		mu_temp[1] = mu[i*2+1];

		meanDiff[iGiD] = data[iGiD] - mu_temp[iGiD%2];

		if(iGiD % 2 == 0){
			pdf[(iGiD/2)*K + i] = (1/sqrt( pow(2*M_PI,dim) * det)) * exp( -0.5* ((meanDiff[iGiD]*covx_inv[0]+meanDiff[iGiD+1]*covx_inv[2])*meanDiff[iGiD] + (meanDiff[iGiD]*covx_inv[1]+meanDiff[iGiD+1]*covx_inv[3])*meanDiff[iGiD+1]));
			//barrier (CLK_GLOBAL_MEM_FENCE);	
			pdf_w[(iGiD/2)*K + i] = pdf[(iGiD/2)*K + i] * phi[i];
		}
	}

	
	for(i=0; i < K; i++){
		if (iGiD % 2 == 0) 
			W[(iGiD/2)*K + i] = pdf_w[(iGiD/2)*K + i] / (pdf_w[(iGiD/2)*K + 0]+ pdf_w[(iGiD/2)*K + 1] + pdf_w[(iGiD/2)*K + 2] + pdf_w[(iGiD/2)*K+3]);
	}

}

//====================================================================================
//M: number of nodes
//K: number of cluster = local size

__kernel void transpose(__global double* W, __global double* WT, 
			int M)
{
	int iGID = get_global_id(0);
	int iLID = get_local_id(0);
	int GroupID = get_group_id(0);
	WT[iLID*M + GroupID] = W[iGID];

}

//calculate summation of one line in the matrix X
//N is the length of that line
//Global: N
//local: arbitrary
__kernel void sum_offset(__global double* X, __global double* scratch, __global double* temp1, 
						__global double* temp2, int N, int N_ext, int index){
	int iGID = get_global_id(0);
	int offset = index*N;
	int i;

	if(iGID < N){
		scratch[iGID] = X[offset+iGID];
	}else{
		scratch[iGID] = 0.0f;
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
	
	if(iGID == 0){
		for(i = 1; i<N; i++){
			scratch[0] += scratch[i];
		}
	
		temp1[index] = scratch[0];
		temp2[index] = scratch[0]/N;
	}
}
//K*M  M*N
__kernel void matrix_multi(__global double* X, __global double* Y,  
			__global double* center_temp,int M, int K, int N)
{
	int k;
	int i = get_global_id(0);
	int j = get_global_id(1);
	double tmp = 0.0f;

	for (k=0; k<M; k=k+1){
		tmp = tmp + X[i*M+k] * Y[k*N+j];
	}
	center_temp[i*N+j] = tmp;
}

//local: N
//global: K*N
__kernel void mu_compute(__global double* multi, __global double* wt_sum,
						__global double* mu,__global double* mu_prev, 
						__global double* diff){
	int iGID = get_global_id(0);
	int GroupID = get_group_id(0);
	int Global_size = get_global_size(0);
	int i;
	mu_prev[iGID] = mu[iGID];
	mu[iGID] = multi[iGID]/wt_sum[GroupID];
	barrier (CLK_GLOBAL_MEM_FENCE);
	if(iGID == 0){
		diff[0] = 0.0f;
		for(i = 0; i<Global_size; i =i+1){
			diff[0] +=fabs(mu[iGID]-mu_prev[iGID]);
		}
	}
	
}

//local:N
//global:K*N
//differnt j needs loop
__kernel void xm_compute_j(__global double* X, __global double* mu, 
							__global double* Xm, int j){
	int iLID = get_local_id(0);
	int iGID = get_global_id(0);
	int Group_size = get_local_size(0);
	Xm[iGID] = X[iGID]-mu[iLID+j*Group_size];
}

//local size: N*N
//global size: M*N*N
//loop by wj
//scratch: N*N*M
__kernel void sigma_k_compute(__global double* XM, 
			 __global double *W, __global double* Sigma_k, __global double* scratch, 
			__global double *WT_sum, int wj, int N, int K)
{
	int iLID = get_local_id(0);
	int iGID = get_global_id(0);
	int GroupID = get_group_id(0);
	int Group_size = get_local_size(0);
	int Group_num = get_num_groups(0);
	int i;
	switch(iLID){
		case(0):
			scratch[iGID] = XM[GroupID*N] * XM[GroupID*N]*W[GroupID*K+wj];
			break;
		case(1):
		case(2):
			scratch[iGID] = XM[GroupID*N] * XM[GroupID*N+1]*W[GroupID*K+wj];
			break;
		case(3):
			scratch[iGID] = XM[GroupID*N+1] * XM[GroupID*N+1]*W[GroupID*K+wj];
			break;
		default:
			scratch[iGID] = 0;
			break;
		
	}
	barrier (CLK_GLOBAL_MEM_FENCE);
	if(GroupID == 0){
		Sigma_k[iGID] = 0.0f;
		for(i = 0; i<Group_num ; i=i+1){
			Sigma_k[iGID] += scratch[i*Group_size+iLID];
		}
		Sigma_k[iGID] = Sigma_k[iGID]/WT_sum[wj];
	}
}





//global:N*N
//loop by wj
__kernel void covar_update(__global double* Sigma, __global double* covx,
							int wj, int N){
	int iGID = get_global_id(0);
	int index = wj*N*N + iGID;
	covx[index] = Sigma[iGID];
}

