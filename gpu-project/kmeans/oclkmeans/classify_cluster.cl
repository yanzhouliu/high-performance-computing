 //data[i*dim]~data[i*(dim+1)-1] --> one node data
 //so is center
 //local size = K
 //global size = K*N
__kernel void find_cluster_center(__global float* data, __global float* center, 
			__global int* C, __global int* Cd,  
			__global float* scratch, __global int* index, int N, int K, int dim, __global int* temp, __global int* prev_C)
{
	
    // get index into global data array
    int iGID = get_global_id(0);
	int iLID = get_local_id(0);
	int offset = get_local_size(0);
	int groupsize = offset;
	int GroupID = get_group_id(0);
	int i = 0;
	float temp_s;

	// distance computation and bound check (equivalent to the limit on a 'for' loop for standard/serial C code)
	barrier (CLK_GLOBAL_MEM_FENCE);
    if (iGID >= N*K)
    {   
        scratch[iGID] = MAXFLOAT;
    }else {

		scratch[iGID] = 0.0;
		for (i = 0; i<dim; i = i+1){
			scratch[iGID] = scratch[iGID] + (data[GroupID*dim+i]-center[iLID*dim+i])*(data[GroupID*dim+i]-center[iLID*dim+i]);
		}
		temp[iGID] = scratch[iGID];
	}
	index[iGID] = iLID;
	
	barrier (CLK_GLOBAL_MEM_FENCE);
	
	for ( offset = offset/2; offset>0; offset >>= 1 ){
		if (iLID < offset){
			scratch[iGID] = (scratch[iGID] < scratch[iGID+offset]) ? scratch[iGID] : scratch[iGID+offset];
			index[iGID] = (scratch[iGID] < scratch[iGID+offset]) ? index[iGID] : index[iGID+offset];
		}
		barrier (CLK_LOCAL_MEM_FENCE);
	}
	barrier (CLK_GLOBAL_MEM_FENCE);
	if (iLID == 0 && GroupID < N){
		C[GroupID] = index[iGID];
	}
	barrier (CLK_GLOBAL_MEM_FENCE);
	if (iGID < K*N){
		Cd[iGID] = 0;
	}
	barrier (CLK_GLOBAL_MEM_FENCE);
	if(GroupID < N){
		Cd[GroupID*K+C[GroupID]] = 1;
	}
	barrier (CLK_GLOBAL_MEM_FENCE);
}

__kernel void compare_center(__global int* C, __global int* temp, __global int* prev_C)
{
	
    // get index into global data array
    int iGID = get_global_id(0);
	int Num = get_global_size(0);
	int diff = 0;
	int i;
	temp[iGID] = (prev_C[iGID] == C[iGID]) ? 0:1;
	barrier (CLK_GLOBAL_MEM_FENCE);
	prev_C[iGID] = C[iGID];
	barrier (CLK_GLOBAL_MEM_FENCE);
	if(iGID == 0){
		for(i = 0; i< Num; i++){
			diff += temp[i];
		}
		temp[0] = diff;
	}
}


//N: number of nodes
//K: number of cluster = local size
//Global Size: >=K*N power of 2
__kernel void cd_transpose(__global int* Cd, __global int* CdT, 
			int N)
{
	int iGID = get_global_id(0);
	int iLID = get_local_id(0);

	int GroupID = get_group_id(0);

	barrier (CLK_GLOBAL_MEM_FENCE);
	CdT[iLID*N + GroupID] = Cd[iGID];
	barrier (CLK_GLOBAL_MEM_FENCE);
}

//i=0~K-1
//j=0~dim-1
__kernel void update_center(__global float* data, __global float* center, 
			 __global int* CdT, int N, int K, int dim, __global float* center_temp, 
			 __global int* conv)
{
	int k;
	int i = get_global_id(0);
	int j = get_global_id(1);
	float tmp = 0.0f;
	int pts = 0;
	float d;
	barrier (CLK_GLOBAL_MEM_FENCE);

	for (k=0; k<N; k=k+1){
		tmp = tmp + CdT[i*N+k] * data[k*dim+j];
		pts = pts + CdT[i*N+k];
		barrier (CLK_GLOBAL_MEM_FENCE);
	}
	tmp = tmp / pts;
	center_temp[i*dim+j] = tmp;
	barrier (CLK_GLOBAL_MEM_FENCE);
	if((i == 0) && (j==0)){
		d = 0.0f;
		for(k = 0; k<K*dim; k = k+1){
			d = d + fabs(center_temp[k] - center[k]);
		}
		if(d<10.0){
			conv = 1;
		}else{
			conv = 0;
		}
	}
	barrier (CLK_GLOBAL_MEM_FENCE);
	center[i*dim+j] = center_temp[i*dim+j];
	barrier (CLK_GLOBAL_MEM_FENCE);
	if((i == 0) && (j==0)){
		center_temp[0] = d;
	}
}



