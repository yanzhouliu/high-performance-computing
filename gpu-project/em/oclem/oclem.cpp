#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

//const char* cSourceFile = "em.cl";
#define matrix_dim 2

// Host buffers for demo
// *********************************************************************
void *X, *W, *WT, *WT_g, *WT_sum, *phi, *multi, *mu, *xm, *sigma, *sigmaS, *covx, *diff;      // Host buffers for OpenCL test


// demo config vars
size_t M = 4000;	//number of nodes
size_t K = 4;	//# of clusters
size_t N = 2;  //dimension
size_t M_ext;
int num;
// OpenCL Vars
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id cdDevice;          // OpenCL device
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel_E;             // OpenCL kernel
cl_kernel ckKernel_T;             // OpenCL kernel
cl_kernel ckKernel_WTS;             // OpenCL kernel  WT_sum
cl_kernel ckKernel_Multi;             // OpenCL kernel  WT_sum
cl_kernel ckKernel_Mu;             // OpenCL kernel  WT_sum
cl_kernel ckKernel_Xmj;             // OpenCL kernel  WT_sum
cl_kernel ckKernel_Sigma;             // OpenCL kernel  WT_sum
cl_kernel ckKernel_CovUpdate;             // OpenCL kernel  WT_sum
//cl_kernel ckKernel_TRSigma;             // OpenCL kernel  WT_sum

cl_event event;

cl_mem cmDevX;
cl_mem cmDevW;
cl_mem cmDevWT;
cl_mem cmDevScratch;
cl_mem cmDevWT_sum;
cl_mem cmDevPHI;
cl_mem cmDevMulti;
cl_mem cmDevMu;
cl_mem cmDevPrevMu;
cl_mem cmDevDiff;
cl_mem cmDevXm;
//cl_mem cmDevTRXm;
cl_mem cmDevSigmaK;
cl_mem cmDevScratchSigma;
cl_mem cmDevCovx;
//=======
cl_mem cmDevCovx_inv;
cl_mem cmDevMeanDiff;
cl_mem cmDevPdf;
cl_mem cmDevPdf_w;
//cl_mem cmDevData;
//cl_mem cmCovx;
// cl_mem cmCovx_inv;
// cl_mem cmMeanDiff;
// cl_mem cmPdf;
// cl_mem cmMu;
// cl_mem cmPhi;
// cl_mem cmPdf_w;
// cl_mem cmW;

//event
cl_event event0;
cl_event event1;
cl_event event2;
cl_event event3;
cl_event event4;
cl_event event5;
cl_event event6;
cl_event event7;

//Expectation
size_t szGlobalWorkSize0;    
size_t szLocalWorkSize0;

//WT
size_t szGlobalWorkSize1;        // 1D var for Total # of work items
size_t szLocalWorkSize1;		    // 1D var for # of work items in the work group
//WTsum
size_t szGlobalWorkSize2;        // 1D var for Total # of work items
size_t szLocalWorkSize2;		    // 1D var for # of work items in the work group
//mu_compute_1
size_t szGlobalWorkSize3[matrix_dim];        // 1D var for Total # of work items
size_t szLocalWorkSize3[matrix_dim];		    // 1D var for # of work items in the work group
//mu_compute_2
size_t szGlobalWorkSize4;        // 1D var for Total # of work items
size_t szLocalWorkSize4;
//xm_compute
size_t szGlobalWorkSize5;        // 1D var for Total # of work items
size_t szLocalWorkSize5;
//sigma_k_sompute
size_t szGlobalWorkSize6;        // 1D var for Total # of work items
size_t szLocalWorkSize6;
//Cov_update
size_t szGlobalWorkSize7;        // 1D var for Total # of work items
size_t szLocalWorkSize7;
//size_t szGlobalWorkSize7[matrix_dim];        // 1D var for Total # of work items
//size_t szLocalWorkSize7[matrix_dim];


size_t szGroupSize;
size_t szParmDataBytes;			// Byte size of context information

size_t szKernelLength;			// Byte size of kernel code
cl_int ciErr1, ciErr2;			// Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
char* cSourceCL = NULL;         // Buffer to hold source for compilation
const char* cExecutableName = NULL;



//Forward Declarations
//*********************************************************************
void Cleanup (int argc, char **argv, int iExitCode);
char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength);
//Main function
//*********************************************************************
int main(int argc, char **argv)
{
	FILE *fp;
	int i,j,k;
	int wi,wj;
	
	cl_ulong time_start, time_end;
	double total_time;

	
	printf("calculate covarience:\n");

    printf("%s Starting...\n\n# of nodes \t= %i\n", argv[0], N);

    // set and log Global and Local work size dimensions
	szLocalWorkSize0 = 2;
    szGlobalWorkSize0 = N*M;  // rounded up to the nearest multiple of the LocalWorkSize
	printf("local0:%d\t global0:%d\n",szLocalWorkSize0, szGlobalWorkSize0);
	
	szLocalWorkSize1 = K;
    szGlobalWorkSize1 = K*M;  // rounded up to the nearest multiple of the LocalWorkSize
	printf("local1:%d\t global1:%d\n",szLocalWorkSize1, szGlobalWorkSize1);
	
	M_ext = exp2(ceil(log2(M)));
    szGlobalWorkSize2 = M_ext;        // 1D var for Total # of work items
	szLocalWorkSize2 = 2;		    // 1D var for # of work items in the work group
	printf("local2:%d\t global2:%d\n",szLocalWorkSize2, szGlobalWorkSize2);
	
	szLocalWorkSize3[0] = M;
	szGlobalWorkSize3[0] = K;
	
	szLocalWorkSize3[1] = M;
	szGlobalWorkSize3[1] = N;
	printf("local3:%d\t%d\t global3:%d\t%d\n",szLocalWorkSize3[0], szLocalWorkSize3[1], szGlobalWorkSize3[0], szGlobalWorkSize3[1]);
	
	szGlobalWorkSize4 = K*N;        // 1D var for Total # of work items
	szLocalWorkSize4 = N;	
	printf("local4:%d\t global4:%d\n",szLocalWorkSize4, szGlobalWorkSize4);
	
	szGlobalWorkSize5 = M*N;        // 1D var for Total # of work items
	szLocalWorkSize5 = N;	
	printf("local5:%d\t global5:%d\n",szLocalWorkSize5, szGlobalWorkSize5);
	
	szLocalWorkSize6 = N*N;
    szGlobalWorkSize6 = N*N*M;  // rounded up to the nearest multiple of the LocalWorkSize
	printf("local6:%d\t global6:%d\n",szLocalWorkSize6, szGlobalWorkSize6);
	
	szLocalWorkSize7 = 2;
    szGlobalWorkSize7 = N*N;  // rounded up to the nearest multiple of the LocalWorkSize
	printf("local7:%d\t global7:%d\n",szLocalWorkSize6, szGlobalWorkSize6);
	
	// Allocate and initialize host arrays
    X = (void *)malloc(sizeof(cl_double) * N * M);
	W = (void *)malloc(sizeof(cl_double) * K * M);
	WT = (void *)malloc(sizeof(cl_double) * K * M);
	WT_g = (void *)malloc(sizeof(cl_double) * K * M);
	WT_sum = (void *)malloc(sizeof(cl_double) * K);
	phi = (void *)malloc(sizeof(cl_double) * K);
	multi = (void *)malloc(sizeof(cl_double) * K *N);
	mu = (void *)malloc(sizeof(cl_double) * K *N);
	xm = (void *)malloc(sizeof(cl_double) * M *N);
	sigma = (void *)malloc(sizeof(cl_double) * N *N);
	sigmaS = (void *)malloc(sizeof(cl_double) * M*N *N);
	covx = (void *)malloc(sizeof(cl_double) * K*N *N);
	diff = (void *)malloc(sizeof(cl_double) * 1);

	if (X == NULL || W == NULL || WT == NULL || WT_g == NULL || WT_sum == NULL){
		printf("malloc failed\n");
		exit(-1);
	}
    //fp = fopen("data.txt","r");
    //fp = fopen("em_1.2k.10000","r");
    fp = fopen("em_3.4k.4000","r");
	for(i = 0; i<N*M; i++){
		fscanf(fp, "%lf",&(((double *)X)[i]));
	}
	fclose(fp);
	
	//fp = fopen("mu.txt","r");
	//fp = fopen("em_1.mu","r");
	fp = fopen("em_3.mu","r");
	for(i = 0; i<K*N; i++){
		fscanf(fp, "%lf",&(((double *)mu)[i]));
	}
	fclose(fp);
	
	//fp = fopen("covx.txt","r");
	//fp = fopen("em_1.covx","r");
	fp = fopen("em_3.covx","r");
	for(i = 0; i<K*N*N; i++){
		fscanf(fp, "%lf",&(((double *)covx)[i]));
	}
	fclose(fp);
	
	for(i = 0; i<K; i++){
		((double *)phi)[i]=1.0/4.0;
	}
	printf("\n\n%f  %f  %f  %f\n\n",((double *)mu)[0],((double *)phi)[0],((double *)X)[0],((double *)covx)[0]);
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

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
	cmDevX = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_double) * N*M, NULL, &ciErr1);;
    ciErr1 |= ciErr2;
	cmDevW = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double) * M * K, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmDevWT = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double) * K * M, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmDevScratch = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double) * M_ext, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmDevWT_sum = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double) * K, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmDevPHI = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double) * K, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmDevMulti = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double) * K *N, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmDevMu = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double) * K *N, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmDevXm = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double) * M *N, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	//cmDevTRXm = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double) * M *N, NULL, &ciErr2);
	//ciErr1 |= ciErr2;
	cmDevSigmaK = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double) * N *N, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmDevScratchSigma = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double) * M * N *N, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmDevCovx = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double) * K * N *N, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	//========================
	cmDevCovx_inv = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double) * N *N, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmDevMeanDiff = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double) * M *N, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmDevPdf = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double) * K * M, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmDevPdf_w = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double) * K * M, NULL, &ciErr2);
	ciErr1 |= ciErr2;	
	cmDevPrevMu = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double) * K * N, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmDevDiff = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double) * 1, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	
	
    printf("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clCreateBuffer, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Read the OpenCL kernel in from source file

	cPathAndName = "./em.cl";
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


    // Create the kernel
	ckKernel_E = clCreateKernel(cpProgram, "expectation", &ciErr1);
	if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clCreateKernel, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
    ckKernel_T = clCreateKernel(cpProgram, "transpose", &ciErr1);
	if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clCreateKernel, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	ckKernel_WTS = clCreateKernel(cpProgram, "sum_offset", &ciErr2);
	ciErr1 |= ciErr2;
    printf("clCreateKernel ...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clCreateKernel, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	ckKernel_Multi = clCreateKernel(cpProgram, "matrix_multi", &ciErr2);
	ciErr1 |= ciErr2;
    printf("clCreateKernel ...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clCreateKernel, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	ckKernel_Mu = clCreateKernel(cpProgram, "mu_compute", &ciErr2);
	ciErr1 |= ciErr2;
    printf("clCreateKernel ...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clCreateKernel, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	ckKernel_Xmj = clCreateKernel(cpProgram, "xm_compute_j", &ciErr2);
	ciErr1 |= ciErr2;
    printf("clCreateKernel ...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clCreateKernel, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

	ckKernel_Sigma = clCreateKernel(cpProgram, "sigma_k_compute", &ciErr1);
	if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clCreateKernel, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	ckKernel_CovUpdate = clCreateKernel(cpProgram, "covar_update", &ciErr1);
	if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clCreateKernel, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	
    // Set the Argument values
	//Expectation:
	ciErr1 = clSetKernelArg(ckKernel_E, 0, sizeof(cl_mem), (void*)&cmDevX);
	ciErr1 |= clSetKernelArg(ckKernel_E, 1, sizeof(cl_mem), (void*)&cmDevCovx);
	ciErr1 |= clSetKernelArg(ckKernel_E, 2, sizeof(cl_mem), (void*)&cmDevMeanDiff);
	ciErr1 |= clSetKernelArg(ckKernel_E, 3, sizeof(cl_mem), (void*)&cmDevPdf);
	ciErr1 |= clSetKernelArg(ckKernel_E, 4, sizeof(cl_mem), (void*)&cmDevMu);
	ciErr1 |= clSetKernelArg(ckKernel_E, 5, sizeof(cl_mem), (void*)&cmDevPHI);
	ciErr1 |= clSetKernelArg(ckKernel_E, 6, sizeof(cl_mem), (void*)&cmDevPdf_w);
	ciErr1 |= clSetKernelArg(ckKernel_E, 7, sizeof(cl_mem), (void*)&cmDevW);
	ciErr1 |= clSetKernelArg(ckKernel_E, 8, sizeof(cl_int), (void*)&M);
	ciErr1 |= clSetKernelArg(ckKernel_E, 9, sizeof(cl_int), (void*)&N);
	ciErr1 |= clSetKernelArg(ckKernel_E, 10, sizeof(cl_int), (void*)&K);
    printf("clSetKernelArg 0 - 10...\n\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clSetKernelArg, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	
    ciErr1 = clSetKernelArg(ckKernel_T, 0, sizeof(cl_mem), (void*)&cmDevW);
	ciErr1 |= clSetKernelArg(ckKernel_T, 1, sizeof(cl_mem), (void*)&cmDevWT);
	ciErr1 |= clSetKernelArg(ckKernel_T, 2, sizeof(cl_int), (void*)&M);
    printf("clSetKernelArg 0 - 2...\n\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clSetKernelArg, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	ciErr1 = clSetKernelArg(ckKernel_WTS, 0, sizeof(cl_mem), (void*)&cmDevWT);
	ciErr1 |= clSetKernelArg(ckKernel_WTS, 1, sizeof(cl_mem), (void*)&cmDevScratch);
	ciErr1 |= clSetKernelArg(ckKernel_WTS, 2, sizeof(cl_mem), (void*)&cmDevWT_sum);
	ciErr1 |= clSetKernelArg(ckKernel_WTS, 3, sizeof(cl_mem), (void*)&cmDevPHI);
	ciErr1 |= clSetKernelArg(ckKernel_WTS, 4, sizeof(cl_int), (void*)&M);
	ciErr1 |= clSetKernelArg(ckKernel_WTS, 5, sizeof(cl_int), (void*)&M_ext);
	//ciErr1 |= clSetKernelArg(ckKernel_WTS, 5, sizeof(cl_int), (void*)&num);
    printf("clSetKernelArg 0 - 5...\n\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clSetKernelArg, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	
	ciErr1 = clSetKernelArg(ckKernel_Multi, 0, sizeof(cl_mem), (void*)&cmDevWT);
	ciErr1 |= clSetKernelArg(ckKernel_Multi, 1, sizeof(cl_mem), (void*)&cmDevX);
	ciErr1 |= clSetKernelArg(ckKernel_Multi, 2, sizeof(cl_mem), (void*)&cmDevMulti);
	ciErr1 |= clSetKernelArg(ckKernel_Multi, 3, sizeof(cl_int), (void*)&M);
	ciErr1 |= clSetKernelArg(ckKernel_Multi, 4, sizeof(cl_int), (void*)&K);
	ciErr1 |= clSetKernelArg(ckKernel_Multi, 5, sizeof(cl_int), (void*)&N);
    printf("clSetKernelArg 0 - 5...\n\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clSetKernelArg, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	ciErr1 = clSetKernelArg(ckKernel_Mu, 0, sizeof(cl_mem), (void*)&cmDevMulti);
	ciErr1 |= clSetKernelArg(ckKernel_Mu, 1, sizeof(cl_mem), (void*)&cmDevWT_sum);
	ciErr1 |= clSetKernelArg(ckKernel_Mu, 2, sizeof(cl_mem), (void*)&cmDevMu);
	ciErr1 |= clSetKernelArg(ckKernel_Mu, 3, sizeof(cl_mem), (void*)&cmDevPrevMu);
	ciErr1 |= clSetKernelArg(ckKernel_Mu, 4, sizeof(cl_mem), (void*)&cmDevDiff);
    printf("clSetKernelArg 0 - 4...\n\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clSetKernelArg, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	ciErr1 = clSetKernelArg(ckKernel_Xmj, 0, sizeof(cl_mem), (void*)&cmDevX);
	ciErr1 |= clSetKernelArg(ckKernel_Xmj, 1, sizeof(cl_mem), (void*)&cmDevMu);
	ciErr1 |= clSetKernelArg(ckKernel_Xmj, 2, sizeof(cl_mem), (void*)&cmDevXm);
	//ciErr1 |= clSetKernelArg(ckKernel_Xmj, 3, sizeof(cl_int), (void*)&j);
    printf("clSetKernelArg 0 - 3...\n\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clSetKernelArg, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	


	ciErr1 = clSetKernelArg(ckKernel_Sigma, 0, sizeof(cl_mem), (void*)&cmDevXm);
	ciErr1 |= clSetKernelArg(ckKernel_Sigma, 1, sizeof(cl_mem), (void*)&cmDevW);
	ciErr1 |= clSetKernelArg(ckKernel_Sigma, 2, sizeof(cl_mem), (void*)&cmDevSigmaK);
	ciErr1 |= clSetKernelArg(ckKernel_Sigma, 3, sizeof(cl_mem), (void*)&cmDevScratchSigma);
	ciErr1 |= clSetKernelArg(ckKernel_Sigma, 4, sizeof(cl_mem), (void*)&cmDevWT_sum);
	//ciErr1 |= clSetKernelArg(ckKernel_Sigma, 5, sizeof(cl_int), (void*)&wj);
	ciErr1 |= clSetKernelArg(ckKernel_Sigma, 6, sizeof(cl_int), (void*)&N);
	ciErr1 |= clSetKernelArg(ckKernel_Sigma, 7, sizeof(cl_int), (void*)&K);
    printf("clSetKernelArg 0 - 7...\n\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clSetKernelArg, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	ciErr1 = clSetKernelArg(ckKernel_CovUpdate, 0, sizeof(cl_mem), (void*)&cmDevSigmaK);
	ciErr1 |= clSetKernelArg(ckKernel_CovUpdate, 1, sizeof(cl_mem), (void*)&cmDevCovx);
	//ciErr1 |= clSetKernelArg(ckKernel_CovUpdate, 2, sizeof(cl_int), (void*)&wj);
	ciErr1 |= clSetKernelArg(ckKernel_CovUpdate, 3, sizeof(cl_int), (void*)&N);
    printf("clSetKernelArg 0 - 3...\n\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clSetKernelArg, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
	
	
    // --------------------------------------------------------
    // Start Core sequence... copy input data to GPU, compute, copy results back

    // Asynchronous write of data to GPU device
    ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, cmDevMu, CL_FALSE, 0, sizeof(cl_double) *K*N, mu, 0, NULL, NULL);
    ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, cmDevX, CL_FALSE, 0, sizeof(cl_double) *N*M, X, 0, NULL, NULL);
    ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, cmDevCovx, CL_FALSE, 0, sizeof(cl_double) *N*N*K, covx, 0, NULL, NULL);
    ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, cmDevPHI, CL_FALSE, 0, sizeof(cl_double) *K, phi, 0, NULL, NULL);

    printf("clEnqueueWriteBuffer (data and center)...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Launch kernel
	// set and log Global and Local work size dimensions

	
	printf("...\n");
	int iter = 0;
	total_time = 0.0;
	do{
		
		ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_E, 1, NULL, &szGlobalWorkSize0, &szLocalWorkSize0, 0, NULL, &event0);
		printf("clEnqueueNDRangeKernel (Expectation)...\n");
		//clFinish(cqCommandQueue);
		//clWaitForEvents(1, &event);
		if (ciErr1 != CL_SUCCESS)
		{
			printf("Error %d in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n",ciErr1, __LINE__, __FILE__);
			Cleanup(argc, argv, EXIT_FAILURE);
		}
		clFinish (cqCommandQueue);	
		clWaitForEvents(1 , &event0);
		clGetEventProfilingInfo(event0, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event0, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		total_time += time_end - time_start;
		printf("%f\n",((time_end - time_start) / 1000000.0));
		
		ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_T, 1, NULL, &szGlobalWorkSize1, &szLocalWorkSize1, 0, NULL, &event1);
		printf("clEnqueueNDRangeKernel (Tanspose)...\n");
		//clFinish(cqCommandQueue);
		//clWaitForEvents(1, &event);
		if (ciErr1 != CL_SUCCESS)
		{
			printf("Error %d in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n",ciErr1, __LINE__, __FILE__);
			Cleanup(argc, argv, EXIT_FAILURE);
		}
		clFinish (cqCommandQueue);
		clWaitForEvents(1 , &event1);
		clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		total_time += time_end - time_start;
		printf("%f\n",((time_end - time_start) / 1000000.0));
		
		printf("WT_summation\n");
		for(num = 0; num<K; num++){
			ciErr1 = clSetKernelArg(ckKernel_WTS, 6, sizeof(cl_int), (void*)&num);
			ciErr1 |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_WTS, 1, NULL, &szGlobalWorkSize2, &szLocalWorkSize2, 0, NULL, &event2);
		}
		if (ciErr1 != CL_SUCCESS)
		{
			printf("Error %d in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n",ciErr1, __LINE__, __FILE__);
			Cleanup(argc, argv, EXIT_FAILURE);
		}
		clFinish (cqCommandQueue);
		clWaitForEvents(1 , &event2);
		clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		total_time += time_end - time_start;
		printf("%f\n",((time_end - time_start) / 1000000.0));
		
		ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_Multi, 2, NULL, szGlobalWorkSize3, NULL, 0, NULL, &event3);
		printf("clEnqueueNDRangeKernel (multi)...\n");
		//clFinish(cqCommandQueue);
		//clWaitForEvents(1, &event);
		if (ciErr1 != CL_SUCCESS)
		{
			printf("Error %d in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n",ciErr1, __LINE__, __FILE__);
			Cleanup(argc, argv, EXIT_FAILURE);
		}
		clFinish (cqCommandQueue);
		clWaitForEvents(1 , &event3);
		clGetEventProfilingInfo(event3, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event3, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		total_time += time_end - time_start;	
		printf("%f\n",((time_end - time_start) / 1000000.0));
		
		printf("clEnqueueNDRangeKernel (beforemu)...\n");
		ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_Mu, 1, NULL, &szGlobalWorkSize4, &szLocalWorkSize4, 0, NULL, &event4);
		printf("clEnqueueNDRangeKernel (mu)...\n");
		//clFinish(cqCommandQueue);
		//clWaitForEvents(1, &event);
		if (ciErr1 != CL_SUCCESS)
		{
			printf("Error %d in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n",ciErr1, __LINE__, __FILE__);
			Cleanup(argc, argv, EXIT_FAILURE);
		}
		clFinish (cqCommandQueue);
		clWaitForEvents(1 , &event4);
		clGetEventProfilingInfo(event4, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event4, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		total_time += time_end - time_start;
		printf("%f\n",((time_end - time_start) / 1000000.0));
		
		for(num = 0; num<K; num++){
			j = num;
			ciErr1 |= clSetKernelArg(ckKernel_Xmj, 3, sizeof(cl_int), (void*)&j);
			ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_Xmj, 1, NULL, &szGlobalWorkSize5, &szLocalWorkSize5, 0, NULL, &event5);
			printf("clEnqueueNDRangeKernel (Xm)...\n");
			//clFinish(cqCommandQueue);
			//clWaitForEvents(1, &event);
			if (ciErr1 != CL_SUCCESS)
			{
				printf("Error %d in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n",ciErr1, __LINE__, __FILE__);
				Cleanup(argc, argv, EXIT_FAILURE);
			}
			clFinish (cqCommandQueue);
			clWaitForEvents(1 , &event5);
			clGetEventProfilingInfo(event5, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
			clGetEventProfilingInfo(event5, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
			total_time += time_end - time_start;
			printf("%f\n",((time_end - time_start) / 1000000.0));
		
			wj = num;
			
			ciErr1 |= clSetKernelArg(ckKernel_Sigma, 5, sizeof(cl_int), (void*)&wj);
			ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_Sigma, 1, NULL, &szGlobalWorkSize6, &szLocalWorkSize6, 0, NULL, &event6);
			printf("clEnqueueNDRangeKernel (Sigma)...\n");
			//clFinish(cqCommandQueue);
			//clWaitForEvents(1, &event);
			if (ciErr1 != CL_SUCCESS)
			{
				printf("Error %d in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n",ciErr1, __LINE__, __FILE__);
				Cleanup(argc, argv, EXIT_FAILURE);
			}
			clFinish (cqCommandQueue);
			clWaitForEvents(1 , &event6);
			clGetEventProfilingInfo(event6, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
			clGetEventProfilingInfo(event6, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
			total_time += time_end - time_start;
		printf("%f\n",((time_end - time_start) / 1000000.0));
		
			wj = num;
			
			ciErr1 |= clSetKernelArg(ckKernel_CovUpdate, 2, sizeof(cl_int), (void*)&wj);
			ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel_CovUpdate, 1, NULL, &szGlobalWorkSize7, &szLocalWorkSize7, 0, NULL, &event7);
			printf("clEnqueueNDRangeKernel (CovUpdate)...\n");
			//clFinish(cqCommandQueue);
			//clWaitForEvents(1, &event);
			if (ciErr1 != CL_SUCCESS)
			{
				printf("Error %d in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n",ciErr1, __LINE__, __FILE__);
				Cleanup(argc, argv, EXIT_FAILURE);
			}
			clFinish (cqCommandQueue);
			clWaitForEvents(1 , &event7);
			clGetEventProfilingInfo(event7, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
			clGetEventProfilingInfo(event7, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
			total_time += time_end - time_start;
		printf("%f\n",((time_end - time_start) / 1000000.0));
		
		}
		ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmDevDiff, CL_TRUE, 0, sizeof(cl_double) * 1, diff, 0, NULL, NULL);
		iter = iter+1;
		if (ciErr1 != CL_SUCCESS)
		{
			printf("Error %d in clEnqueueReadBuffer, Line %u in file %s!!!\n\n", ciErr1, __LINE__, __FILE__);
			Cleanup(argc, argv, EXIT_FAILURE);
		}
		printf("iter:%d diff:%f\n\n",iter, ((double *)diff)[0]);
		printf("execution time:%.6f ms\n\n",iter, (total_time / 1000000.0));
	}while((((double *)diff)[0] > 0.24067 && iter < 1000) || iter <108);
	
	printf("iter:%d diff:%f\n\n",iter, ((double *)diff)[0]);
	printf("execution time:%.6f ms\n\n",iter, (total_time / 1000000.0));
    // Synchronous/blocking read of results, and check accumulated errors
	
	ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmDevW, CL_TRUE, 0, sizeof(cl_double) * K*M, W, 0, NULL, NULL);
	ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmDevWT, CL_TRUE, 0, sizeof(cl_double) * K*M, WT_g, 0, NULL, NULL);
	ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmDevWT_sum, CL_TRUE, 0, sizeof(cl_double) * K, WT_sum, 0, NULL, NULL);
	ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmDevPHI, CL_TRUE, 0, sizeof(cl_double) * K, phi, 0, NULL, NULL);
	ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmDevMulti, CL_TRUE, 0, sizeof(cl_double) * K*N, multi, 0, NULL, NULL);
	ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmDevMu, CL_TRUE, 0, sizeof(cl_double) * K*N, mu, 0, NULL, NULL);
	ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmDevXm, CL_TRUE, 0, sizeof(cl_double) * M*N, xm, 0, NULL, NULL);
	//ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmDevTRXm, CL_TRUE, 0, sizeof(cl_double) * M*N, xm, 0, NULL, NULL);
	//ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmDevScratchSigma, CL_TRUE, 0, sizeof(cl_double) * M*N*N, sigmaS, 0, NULL, NULL);
	ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmDevSigmaK, CL_TRUE, 0, sizeof(cl_double) * N*N, sigmaS, 0, NULL, NULL);
	ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmDevCovx, CL_TRUE, 0, sizeof(cl_double) * K*N*N, covx, 0, NULL, NULL);
	//ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmDevDiff, CL_TRUE, 0, sizeof(cl_double) * 1, diff, 0, NULL, NULL);
	printf("clEnqueueReadBuffer (WT)...\n\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clEnqueueReadBuffer, Line %u in file %s!!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    //--------------------------------------------------------
	printf("printing\n");
    // Compute and compare results for golden-host and report errors and pass/fail
	fp = fopen("W_g.txt","w");
	for(i=0; i<K*M; i++){
		
		fprintf(fp,"%f\t",((double *)W)[i]);
		if(i%K == K-1)
			fprintf(fp,"\n");
	}
	fp = fopen("WT_g.txt","w");
	for(i=0; i<K*M; i++){
		if(((double *)WT)[i] - ((double *)WT_g)[i] > 0.0001){
			printf("incorrect\n");
			break;
		}
		fprintf(fp,"%f\t",((double *)WT_g)[i]);
		if(i%M == M-1)
			fprintf(fp,"\n");
	}
	double result = 0;
	for(i = 0; i<K; i++){
		result = 0;
		for(j = 0; j<M; j++){
			result += ((double *)WT_g)[i*M+j];
		}
		printf("%f\t",result);
	}
	printf("\n");
	for(i=0; i<K; i++){
		printf("%f\t",((double *)phi)[i]);
		printf("%f\n",((double *)WT_sum)[i]);
		
			
	}
	printf("\n");
	printf("multi:\n");
	for(i=0; i<K*N; i++){
		printf("%f\t",((double *)multi)[i]);
		if(i%N == N-1){
			printf("\n");
		}
			
	}
	printf("\n");
	printf("mu:\n");
	for(i=0; i<K*N; i++){
		printf("%f\t",((double *)mu)[i]);
		if(i%N == N-1){
			printf("\n");
		}
			
	}
	printf("\n");
	printf("covx:\n");
	for(i=0; i<K*N*N; i++){
		printf("%f\t",((double *)covx)[i]);
		if(i%N == N-1){
			printf("\n");
		}
		if(i%(N*N) == N*N-1){
			printf("\n");
		}
			
	}
	printf("\n");
	
	fp = fopen("Xm.txt","w");
	for(i=0; i<M*N; i++){
		fprintf(fp,"%f\t",((double *)xm)[i]);
		if(i%N == N-1){
			fprintf(fp,"\n");
		}
	}
	fclose(fp);
	
	fp = fopen("SigmaS.txt","w");
	for(i=0; i<N*N; i++){
		fprintf(fp,"%f\t",((double *)sigmaS)[i]);
		if(i%N == N-1){
			fprintf(fp,"\n");
		}
		
		if(i%(N*N) == N*N-1){
			fprintf(fp,"\n");
		}
		
	}
	fclose(fp);
	
	ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmDevScratch, CL_TRUE, 0, sizeof(cl_double) * M_ext, WT_g, 0, NULL, NULL);
	fp = fopen("WT_scratch.txt","w");
	for(i=0; i<M_ext; i++){
		fprintf(fp,"%f\t",((double *)WT_g)[i]);
	}
	fclose(fp);
	Cleanup (argc, argv, 0);

    printf("end\n");
	return 0;
}

void Cleanup (int argc, char **argv, int iExitCode)
{

	if(ckKernel_E)clReleaseKernel(ckKernel_E);
	if(ckKernel_T)clReleaseKernel(ckKernel_T);
	if(ckKernel_WTS)clReleaseKernel(ckKernel_WTS);
	if(ckKernel_Multi)clReleaseKernel(ckKernel_Multi);
	if(ckKernel_Mu)clReleaseKernel(ckKernel_Mu);
	if(ckKernel_Xmj)clReleaseKernel(ckKernel_Xmj);
	if(ckKernel_Sigma)clReleaseKernel(ckKernel_Sigma);
	if(ckKernel_CovUpdate)clReleaseKernel(ckKernel_CovUpdate);
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);

	if(cmDevW)clReleaseMemObject(cmDevW);
	if(cmDevWT)clReleaseMemObject(cmDevWT);
	if(cmDevWT_sum)clReleaseMemObject(cmDevWT_sum);
	if(cmDevScratch)clReleaseMemObject(cmDevScratch);
	if(cmDevPHI)clReleaseMemObject(cmDevPHI);
	if(cmDevMulti)clReleaseMemObject(cmDevMulti);
	if(cmDevMu)clReleaseMemObject(cmDevMu);
	if(cmDevXm)clReleaseMemObject(cmDevXm);
	if(cmDevSigmaK)clReleaseMemObject(cmDevSigmaK);
	if(cmDevScratchSigma)clReleaseMemObject(cmDevScratchSigma);
	if(cmDevCovx)clReleaseMemObject(cmDevCovx);
	//=======================
	if(cmDevCovx_inv)clReleaseMemObject(cmDevCovx_inv);
	if(cmDevMeanDiff)clReleaseMemObject(cmDevMeanDiff);
	if(cmDevPdf)clReleaseMemObject(cmDevPdf);
	if(cmDevPdf_w)clReleaseMemObject(cmDevPdf_w);
	if(cmDevPdf_w)clReleaseMemObject(cmDevPrevMu);
	if(cmDevPdf_w)clReleaseMemObject(cmDevDiff);
	
    // Free host memory
    free(X);
	free(W);
	free(WT);
	free(WT_g);
	free(WT_sum);
	free(multi);
	free(mu);
	free(xm);
	free(sigma);
	free(sigmaS);
	free(covx);
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
