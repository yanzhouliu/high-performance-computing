#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <CL/opencl.h>
#include <oclUtils.h>
#include <shrQATest.h>
#include <time.h>

#define matrix_dim 2
const char* cSourceFile = "classify_cluster.cl";

// Host buffers for demo
// *********************************************************************
void *data, *center, *C, *Cd, *CdT, *temp_k, *center_temp, *zero_pad;        // Host buffers for OpenCL test
int Conv;
float diff;
int C_diff;
// OpenCL Vars
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id cdDevice;          // OpenCL device
cl_program cpProgram;           // OpenCL program

cl_kernel ckKernel_find;             // OpenCL kernel
cl_kernel ckKernel_tr;             // OpenCL kernel
cl_kernel ckKernel_update;             // OpenCL kernel
cl_kernel ckKernel_cmp;             // OpenCL kernel

cl_event event;
cl_event event1;
cl_event event2;
cl_event event3;


cl_mem cmDevData;               
cl_mem cmDevCenter;       	//previous_center 
cl_mem cmDevC;            
cl_mem cmDevCd;          
cl_mem cmDevScratch;    
cl_mem cmDevIndex;  
cl_mem cmDevCdT;   
cl_mem cmCenter_temp;	//new_center
cl_mem cmDevConv;
cl_mem cmDevDiff;
cl_mem cmDevPrev_C;

size_t szGlobalWorkSize_K1;        // 1D var for Total # of work items
size_t szLocalWorkSize_K1;		    // 1D var for # of work items in the work group	
size_t szGroupSize_K1;

size_t szGlobalWorkSize_K2;        // 1D var for Total # of work items
size_t szLocalWorkSize_K2;		    // 1D var for # of work items in the work group	
size_t szGroupSize_K2;

size_t szGlobalWorkSize_K3[matrix_dim];        // 1D var for Total # of work items
size_t szLocalWorkSize_K3[matrix_dim];		    // 1D var for # of work items in the work group
size_t szGroupSize_K3;

size_t szGlobalWorkSize_K4;        // 1D var for Total # of work items
size_t szLocalWorkSize_K4;		    // 1D var for # of work items in the work group	


size_t szParmDataBytes;			// Byte size of context information

size_t szKernelLength;			// Byte size of kernel code
cl_int ciErr1, ciErr2;			// Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
char* cSourceCL = NULL;         // Buffer to hold source for compilation
const char* cExecutableName = NULL;

// demo config vars
int N = 4000;	//number of nodes
int	K = 4;		//number of clusters
int dim = 2;	//number of dimensions

//Forward Declarations
//*********************************************************************
void Cleanup (int argc, char **argv, int iExitCode);
char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength);
//Main function 
//*********************************************************************
int main(int argc, char **argv)
{
	FILE *fp;
	int i,j;
	int iter = 0;
	
	clock_t begin, end;
	double time_spent;
	cl_ulong time_start, time_end;
	double total_time;
	printf("kmeans:\n");
	
    printf("%s Starting...\n\n# of nodes \t= %i\n", argv[0], N); 

    // set and log Global and Local work size dimensions
    szLocalWorkSize_K1 = K;
    szGlobalWorkSize_K1 = ((size_t)((float)(N*K/szLocalWorkSize_K1)+0.5))*szLocalWorkSize_K1;  // rounded up to the nearest multiple of the LocalWorkSize
  	szGroupSize_K1 = szGlobalWorkSize_K1/szLocalWorkSize_K1;
	
	szLocalWorkSize_K4 = 2;
	//szGroupSize_K4 = szGroupSize_K1;
	szGlobalWorkSize_K4 = N;
	
	szLocalWorkSize_K2 = szLocalWorkSize_K1;
	szGroupSize_K2 = szGroupSize_K1;
	szGlobalWorkSize_K2 = szGlobalWorkSize_K1;
	
	szLocalWorkSize_K3[0] = N;
	szGlobalWorkSize_K3[0] = K;
	
	szLocalWorkSize_K3[1] = N;
	szGlobalWorkSize_K3[1] = dim;
	
	printf("K1: global work size: %d; local size:%d; group size:%d\n", szGlobalWorkSize_K1, szLocalWorkSize_K1, szGroupSize_K1);
	printf("K4: global work size: %d; local size:%d; group size:%d\n", szGlobalWorkSize_K4, szLocalWorkSize_K4);
	printf("K2: global work size: %d; local size:%d; group size:%d\n", szGlobalWorkSize_K2, szLocalWorkSize_K2, szGroupSize_K2);
	printf("K3: szLocalWorkSize_K3[0]: %d; szGlobalWorkSize_K3[0]:%d\n", szLocalWorkSize_K3[0], szGlobalWorkSize_K3[0]);
	printf("K3: szLocalWorkSize_K3[1]: %d; szGlobalWorkSize_K3[1]:%d\n", szLocalWorkSize_K3[1], szGlobalWorkSize_K3[1]);
   
    // Allocate and initialize host arrays 
    //Kernel 1 find_cluster_center
	data = (void *)malloc(sizeof(cl_float) * szGroupSize_K1*dim);
    center = (void *)malloc(sizeof(cl_float) * szLocalWorkSize_K1*dim);
    C = (void *)malloc(sizeof(cl_int) * szGroupSize_K1);
	temp_k = (void *)malloc(sizeof(cl_float) * szGlobalWorkSize_K1);
	Cd = (void *)malloc(sizeof(cl_int) * szGlobalWorkSize_K1);
	
	//Kernel 2 Transpose
	CdT = (void *)malloc(sizeof(cl_int) * szGlobalWorkSize_K2);
	
	//Kernel 3 Update_center
	center_temp = (void *)malloc(sizeof(cl_float) * K *dim);
	zero_pad = (void *)malloc(sizeof(cl_float) * K *dim);
	
	for(i = 0; i<K*dim; i++){
		(((float *)zero_pad)[i]) = 0.0;
	}
	
	if (data == NULL || center == NULL || C==NULL){
		printf("malloc failed\n");
		exit(-1);
	}

    fp = fopen("kmeans_1.4k.2d.4000","r");
	if(fp==NULL){
		printf("kmeans failed\n");
	}
	for(i = 0; i<N*dim; i++){
		fscanf(fp, "%f",&(((float *)data)[i]));
	}
	for(i = 0; i<szGroupSize_K1; i++){
		((int *)C)[i] = 0;
	}
	
	fclose(fp);

	fp = fopen("kmeans_1.center","r");

	if(fp==NULL){
		printf("cen failed\n");
	}
	for(i = 0; i<K*dim; i++)
		fscanf(fp, "%f",&(((float *)center)[i]));
	fclose(fp);
    printf("data:%f  center:%f\n", (((float *)data)[i]), (((float *)center)[i]));
	//Get an OpenCL platform
    ciErr1 = clGetPlatformIDs(1, &cpPlatform, NULL);
    printf("clGetPlatformID...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clGetPlatformID, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    //Get the devices
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
    printf("clGetDeviceIDs...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clGetDeviceIDs, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    //Create the context
    cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
    printf("clCreateContext...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clCreateContext, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Create a command-queue
    //cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, CL_QUEUE_PROFILING_ENABLE, &ciErr1);
    printf("clCreateCommandQueue...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clCreateCommandQueue, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

	cl_ulong size;
	clGetDeviceInfo(cdDevice, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
	printf("\nlocal size:%d\n",size);
	
    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
	//Kernel 1:find_cluster_center
	cmDevData = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float) * szGroupSize_K1*dim, NULL, &ciErr1);
    cmDevCenter = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float) * szLocalWorkSize_K1*dim, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmDevC = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_int) * szGroupSize_K1, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmDevCd = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_int) * szGlobalWorkSize_K1, NULL, &ciErr2);
    ciErr1 |= ciErr2;
	cmDevScratch = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float) * szGlobalWorkSize_K1, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmDevIndex = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_int) * szGlobalWorkSize_K1, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmDevPrev_C = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_int) * szGlobalWorkSize_K1, NULL, &ciErr2);
    ciErr1 |= ciErr2;
	cl_mem temp = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_int) * szGlobalWorkSize_K1, NULL, &ciErr2);
    ciErr1 |= ciErr2;
	
	//Kernel2: Transpose
    cl_mem cmDevCdT = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_int) * szGlobalWorkSize_K2, NULL, &ciErr2);
    ciErr1 |= ciErr2;
	
	//Kernel3: Update_center
	cmCenter_temp = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float) * K * dim, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cl_mem cmDevConv = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_int) * 1, NULL, &ciErr2);
    ciErr1 |= ciErr2;
	cl_mem cmDevDiff = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float) * 1, NULL, &ciErr2);
    ciErr1 |= ciErr2;
	
	printf("clCreateBuffer...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clCreateBuffer, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    
    // Read the OpenCL kernel in from source file
	cPathAndName = "../../../src/oclkmeans/classify_cluster.cl";
    cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength);

    // Create the program
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErr1);
    printf("clCreateProgramWithSource...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clCreateProgramWithSource, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Build the program with 'mad' Optimization option
    #ifdef MAC
        char* flags = "-cl-fast-relaxed-math -DMAC";
    #else
        char* flags = "-cl-fast-relaxed-math";
    #endif
    ciErr1 = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    
    printf("clBuildProgram...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        
        printf("Error %d in clBuildProgram, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Create the kernel--find-cluster-center
    ckKernel_find = clCreateKernel(cpProgram, "find_cluster_center", &ciErr1);
    
    printf("clCreateKernel (find_cluster_center)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        
        printf("Error %d in clCreateKernel, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	// Create the kernel--find-cluster-center
    ckKernel_tr = clCreateKernel(cpProgram, "cd_transpose", &ciErr1);
    
    printf("clCreateKernel (cd_transpose)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        
        printf("Error %d in clCreateKernel, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	// Create the kernel--find-cluster-center
    ckKernel_update = clCreateKernel(cpProgram, "update_center", &ciErr1);
    
    printf("clCreateKernel (update_center)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        
        printf("Error %d in clCreateKernel, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	// Create the kernel--find-cluster-center
    ckKernel_cmp = clCreateKernel(cpProgram, "compare_center", &ciErr1);
    printf("clCreateKernel (compare_center)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {   
        printf("Error %d in clCreateKernel, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Set the Argument values
	//Kernel 1: find_cluster_center
    ciErr1 = clSetKernelArg(ckKernel_find, 0, sizeof(cl_mem), (void*)&cmDevData);
    ciErr1 |= clSetKernelArg(ckKernel_find, 1, sizeof(cl_mem), (void*)&cmDevCenter);
    ciErr1 |= clSetKernelArg(ckKernel_find, 2, sizeof(cl_mem), (void*)&cmDevC);
    ciErr1 |= clSetKernelArg(ckKernel_find, 3, sizeof(cl_mem), (void*)&cmDevCd);
    ciErr1 |= clSetKernelArg(ckKernel_find, 4, sizeof(cl_mem), (void*)&cmDevScratch);
    ciErr1 |= clSetKernelArg(ckKernel_find, 5, sizeof(cl_mem), (void*)&cmDevIndex);
    ciErr1 |= clSetKernelArg(ckKernel_find, 6, sizeof(cl_int), (void*)&N);
    ciErr1 |= clSetKernelArg(ckKernel_find, 7, sizeof(cl_int), (void*)&K);
    ciErr1 |= clSetKernelArg(ckKernel_find, 8, sizeof(cl_int), (void*)&dim);
    ciErr1 |= clSetKernelArg(ckKernel_find, 9, sizeof(cl_mem), (void*)&temp);
    ciErr1 |= clSetKernelArg(ckKernel_find, 10, sizeof(cl_mem), (void*)&cmDevPrev_C);

    printf("clSetKernelArg 0 - 10...\n\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clSetKernelArg, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	//Kernel 2: Transpose
    ciErr1 = clSetKernelArg(ckKernel_tr, 0, sizeof(cl_mem), (void*)&cmDevCd);
    ciErr1 |= clSetKernelArg(ckKernel_tr, 1, sizeof(cl_mem), (void*)&cmDevCdT);
    ciErr1 |= clSetKernelArg(ckKernel_tr, 2, sizeof(cl_int), (void*)&N);
    
    printf("clSetKernelArg 0 - 2...\n\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clSetKernelArg, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	//Kernel 3: update_center
	ciErr1 = clSetKernelArg(ckKernel_update, 0, sizeof(cl_mem), (void*)&cmDevData);
    ciErr1 |= clSetKernelArg(ckKernel_update, 1, sizeof(cl_mem), (void*)&cmDevCenter);
    ciErr1 |= clSetKernelArg(ckKernel_update, 2, sizeof(cl_mem), (void*)&cmDevCdT);
    ciErr1 |= clSetKernelArg(ckKernel_update, 3, sizeof(cl_int), (void*)&N);
    ciErr1 |= clSetKernelArg(ckKernel_update, 4, sizeof(cl_int), (void*)&K);
    ciErr1 |= clSetKernelArg(ckKernel_update, 5, sizeof(cl_int), (void*)&dim);
	ciErr1 |= clSetKernelArg(ckKernel_update, 6, sizeof(cl_mem), (void*)&cmCenter_temp);
	ciErr1 |= clSetKernelArg(ckKernel_update, 7, sizeof(cl_mem), (void*)&cmDevConv);
	
    printf("clSetKernelArg 0 - 7...\n\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clSetKernelArg, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	//Kernel 4: Compare_center
    ciErr1 = clSetKernelArg(ckKernel_cmp, 0, sizeof(cl_mem), (void*)&cmDevC);
    ciErr1 |= clSetKernelArg(ckKernel_cmp, 1, sizeof(cl_mem), (void*)&temp);
    ciErr1 |= clSetKernelArg(ckKernel_cmp, 2, sizeof(cl_mem), (void*)&cmDevPrev_C);
    
    printf("clSetKernelArg 0 - 2...\n\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clSetKernelArg, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	
    // --------------------------------------------------------
    // Start Core sequence... copy input data to GPU, compute, copy results back

    // Asynchronous write of data to GPU device
    ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, cmDevData, CL_FALSE, 0, sizeof(cl_float) * szGroupSize_K1*dim, data, 0, NULL, NULL);
    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmDevCenter, CL_FALSE, 0, sizeof(cl_float) * szLocalWorkSize_K1*dim, center, 0, NULL, NULL);
    
	ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmCenter_temp, CL_FALSE, 0, sizeof(cl_float) * szLocalWorkSize_K1*dim, zero_pad, 0, NULL, NULL);
	ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmDevC, CL_FALSE, 0, sizeof(cl_int) * szGroupSize_K1, C, 0, NULL, NULL);
    clFinish (cqCommandQueue);
	
	
	printf("clEnqueueWriteBuffer (data and center)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Launch kernel
	begin = clock();
	iter =0;
	diff = 1.0;
	
	do{
		ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_find, 1, NULL, &szGlobalWorkSize_K1, &szLocalWorkSize_K1, 0, NULL, &event1);		
		printf("clEnqueueNDRangeKernel (find_cluster_center)...\n"); 
		if (ciErr1 != CL_SUCCESS)
		{
			printf("Error %d in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n",ciErr1, __LINE__, __FILE__);
			Cleanup(argc, argv, EXIT_FAILURE);
		}
		clFinish (cqCommandQueue);
		
		
		ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_cmp, 1, NULL, &szGlobalWorkSize_K4, &szLocalWorkSize_K4, 0, NULL, &event1);		
		printf("clEnqueueNDRangeKernel (find_cluster_center)...\n"); 
		if (ciErr1 != CL_SUCCESS)
		{
			printf("Error %d in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n",ciErr1, __LINE__, __FILE__);
			Cleanup(argc, argv, EXIT_FAILURE);
		}
		clFinish (cqCommandQueue);
		
		
		ciErr1 = clEnqueueReadBuffer(cqCommandQueue, temp, CL_TRUE, 0, sizeof(cl_int) * 1, &C_diff, 0, NULL, NULL);
		clFinish (cqCommandQueue);
		
		ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_tr, 1, NULL, &szGlobalWorkSize_K2, &szLocalWorkSize_K2, 0, NULL, &event2);
		printf("clEnqueueNDRangeKernel (Transpose)...\n"); 
		if (ciErr1 != CL_SUCCESS)
		{
			printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
			Cleanup(argc, argv, EXIT_FAILURE);
		}
		clFinish (cqCommandQueue);

		ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_update, 2, NULL, szGlobalWorkSize_K3, NULL, 0, NULL, &event3);
		printf("clEnqueueNDRangeKernel (update_center)...\n"); 
		if (ciErr1 != CL_SUCCESS)
		{
			printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
			Cleanup(argc, argv, EXIT_FAILURE);
		}
		clFinish (cqCommandQueue);
		ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmCenter_temp, CL_TRUE, 0, sizeof(cl_float) * 1, &diff, 0, NULL, NULL);
		clFinish (cqCommandQueue);
		
		iter++;
		printf("iter:%d\t  diff:%f\t C_diff:%d\n",iter,diff, C_diff);
	}while (diff > 0 && iter < 100);
	end = clock();
	time_spent = (double)(end - begin)/ CLOCKS_PER_SEC*1000000;
	
	printf("time:%.16lf\n",time_spent);
	
	clWaitForEvents(1 , &event1);
	clWaitForEvents(1 , &event2);
	clWaitForEvents(1 , &event3);
	
	clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	printf("\nExecution time 1 in milliseconds = %0.3f ms\n", (total_time / 1000000.0) );

	clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	printf("\nExecution time 2 in milliseconds = %0.3f ms\n", (total_time / 1000000.0) );
	
	clGetEventProfilingInfo(event3, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event3, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	printf("\nExecution time 3 in milliseconds = %0.3f ms\n", (total_time / 1000000.0) );

	
    // Synchronous/blocking read of results, and check accumulated errors
    ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmDevC, CL_TRUE, 0, sizeof(cl_int) * szGroupSize_K1, C, 0, NULL, NULL);
	ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, temp, CL_TRUE, 0, sizeof(cl_int) * szGlobalWorkSize_K1, temp_k, 0, NULL, NULL);
    ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmDevCd, CL_TRUE, 0, sizeof(cl_int) * szGlobalWorkSize_K1, Cd, 0, NULL, NULL);
    ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmDevCdT, CL_TRUE, 0, sizeof(cl_int) * szGlobalWorkSize_K2, CdT, 0, NULL, NULL);
    ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmDevCenter, CL_TRUE, 0, sizeof(cl_float) * K*dim, center_temp, 0, NULL, NULL);
    
	clFinish (cqCommandQueue);
	
	printf("clEnqueueReadBuffer (C)...\n\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clEnqueueReadBuffer, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    //--------------------------------------------------------

    fp = fopen("C.txt","w");
	for(i = 0; i<N; i++){
		fprintf(fp, "%d\n",(((int *)C)[i]));
	}
	fclose(fp);
	
	fp = fopen("d_dim.txt","w");
	for(i = 0; i<K*N; i++){
		fprintf(fp, "%d\n",(((int *)temp_k)[i]));
	}
	fclose(fp);
	
	fp = fopen("Cd.txt","w");
	for(i = 0; i<K*N; i++){
		fprintf(fp, "%d\t",(((int *)Cd)[i]));
		if (i%K == K-1)
			fprintf(fp, "\n");
	}
	fclose(fp);
	
	fp = fopen("CdT.txt","w");
	for(i = 0; i<K*N; i++){
		fprintf(fp, "%d\t",(((int *)CdT)[i]));
		if ((i+1)%N == 0)
			fprintf(fp, "\n");
	}
	fclose(fp);
	
	fp = fopen("new_C.txt","w");
	for(i = 0; i<K*dim; i++){
		fprintf(fp, "%f\t",(((float *)center_temp)[i]));
		if ((i+1)%dim == 0)
			fprintf(fp, "\n");
	}
	fclose(fp);
	printf("iter:%d\n",iter);
	
	Cleanup (argc, argv, 0);
    printf("end\n");
}

void Cleanup (int argc, char **argv, int iExitCode)
{

	if(ckKernel_find)clReleaseKernel(ckKernel_find);  
	if(ckKernel_tr)clReleaseKernel(ckKernel_tr);  
	if(ckKernel_update)clReleaseKernel(ckKernel_update);  
	if(ckKernel_cmp)clReleaseKernel(ckKernel_cmp);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    
	if(cmDevData)clReleaseMemObject(cmDevData);
    if(cmDevCenter)clReleaseMemObject(cmDevCenter);
    if(cmDevC)clReleaseMemObject(cmDevC);
	if(cmDevCd)clReleaseMemObject(cmDevCd);
    if(cmDevScratch)clReleaseMemObject(cmDevScratch);
    if(cmDevIndex)clReleaseMemObject(cmDevIndex);
    if(cmDevCdT)clReleaseMemObject(cmDevCdT);
    if(cmCenter_temp)clReleaseMemObject(cmCenter_temp);
		
    // Free host memory
    free(data); 
    free(center);
    free (C);
	free(Cd);
	free(CdT);
	free(center_temp);
    // finalize logs and leave
	printf("clean up end\n");
}

char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength)
{
    // locals 
    FILE* pFileStream = NULL;
    size_t szSourceLength;
	size_t szPreambleLength;
    // open the OpenCL source code file
    #ifdef _WIN32   // Windows version
        if(fopen_s(&pFileStream, cFilename, "rb") != 0) 
        {       
            return NULL;
        }
    #else           // Linux version
        pFileStream = fopen(cFilename, "rb");
        if(pFileStream == 0) 
        {       
            return NULL;
        }
    #endif

    szPreambleLength = strlen(cPreamble);

    // get the length of the source code
    fseek(pFileStream, 0, SEEK_END); 
    szSourceLength = ftell(pFileStream);
    fseek(pFileStream, 0, SEEK_SET); 

    // allocate a buffer for the source code string and read it in
    char* cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1); 
    memcpy(cSourceString, cPreamble, szPreambleLength);
    if (fread((cSourceString) + szPreambleLength, szSourceLength, 1, pFileStream) != 1)
    {
        fclose(pFileStream);
        free(cSourceString);
        return 0;
    }

    // close the file and return the total length of the combined (preamble + source) string
    fclose(pFileStream);
    if(szFinalLength != 0)
    {
        *szFinalLength = szSourceLength + szPreambleLength;
    }
    cSourceString[szSourceLength + szPreambleLength] = '\0';
	if (cSourceString == NULL)
		printf("null string\n");
    return cSourceString;
}
