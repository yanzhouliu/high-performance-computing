#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/opencl.h>

//const char* cSourceFile = "update_center.cl";
#define matrix_dim 1

// Host buffers for demo
// *********************************************************************
void *data;      // Host buffers for OpenCL test
void *covx, *covx_inv, *meanDiff, *pdf, *mu, *phi, *pdf_w, *W;

// demo config vars
size_t N = 1000;	//number of nodes
size_t K = 3;		//number of clusters
size_t dim = 2;	//number of dimensions

// OpenCL Vars
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id cdDevice;          // OpenCL device
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel;             // OpenCL kernel

cl_mem cmDevData;
cl_mem cmCovx;
cl_mem cmCovx_inv;
cl_mem cmMeanDiff;
cl_mem cmPdf;
cl_mem cmMu;
cl_mem cmPhi;
cl_mem cmPdf_w;
cl_mem cmW;

size_t szGlobalWorkSize[matrix_dim];        // 1D var for Total # of work items
size_t szLocalWorkSize[matrix_dim];		    // 1D var for # of work items in the work group
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
	int i;

	printf("calculate covarience:\n");

    printf("%s Starting...\n\n# of nodes \t= %i\n", argv[0], N);

    // set and log Global and Local work size dimensions
	szLocalWorkSize[0] = dim;
    szGlobalWorkSize[0] = N*dim;  // rounded up to the nearest multiple of the LocalWorkSize

	printf("global work size: %d; local size:%d;\n ", szGlobalWorkSize[0], szLocalWorkSize[0]);

    // Allocate and initialize host arrays
    data = (void *)malloc(sizeof(cl_float) * N *dim);
	covx = (void *)malloc(sizeof(cl_float) * dim * dim * K);
	covx_inv = (void *)malloc(sizeof(cl_float) * dim *dim);
	meanDiff = (void *)malloc(sizeof(cl_float) * N *dim);
	mu = (void *)malloc(sizeof(cl_float) * K *dim);
	pdf = (void *)malloc(sizeof(cl_float) * N * K);
	phi = (void *)malloc(sizeof(cl_float) * K);
	pdf_w = (void *)malloc(sizeof(cl_float) * K * N);
	W = (void *)malloc(sizeof(cl_float) * K * N);

	if (data == NULL || covx == NULL || covx_inv == NULL || meanDiff == NULL || pdf == NULL || phi == NULL){
		printf("malloc failed\n");
		exit(-1);
	}


	((float *)phi)[0] = 1/(float)3;
	((float *)phi)[1] = 1/(float)3;
	((float *)phi)[2] = 1/(float)3;
	printf("%f\t",((float *)phi)[0] );
		
	fp = fopen("data.txt","r");
	for(i = 0; i<N*dim; i++){
		fscanf(fp, "%f",&(((float *)data)[i]));
		//printf("%f\t", (((float *)data)[i]));
		//if(i%2==1) printf("\n");
	}
	fclose(fp);

	fp = fopen("mu.txt","r");
	printf("\n");
	for(i = 0; i<K*dim; i++){
		fscanf(fp, "%f",&(((float *)mu)[i]));
		printf("%f\t", (((float *)data)[i]));
		if(i%2==1) printf("\n");
	}
	fclose(fp);

	fp = fopen("covx.txt","r");
	for(i = 0; i<K*dim*dim; i++){
		fscanf(fp, "%f",&(((float *)covx)[i]));
		printf("%f\t", (((float *)covx)[i]));
		if(i%2==1) printf("\n");
	}
	fclose(fp);

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
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
    printf("clCreateCommandQueue...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clCreateCommandQueue, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
	cmDevData = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * N*dim, NULL, &ciErr1);;
    ciErr1 |= ciErr2;
	cmCovx = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float) * dim * dim * K, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmCovx_inv = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float) * dim * dim, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmMeanDiff = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float) * N * dim, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmPdf = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float) * N * K, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmMu = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float) * dim * K, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmPhi = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float) * K, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmPdf_w = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float) * K * N, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmW = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float) * K * N, NULL, &ciErr2);
	ciErr1 |= ciErr2;

    printf("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clCreateBuffer, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Read the OpenCL kernel in from source file

	cPathAndName = "./update_center.cl";
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

    ckKernel = clCreateKernel(cpProgram, "update_center", &ciErr1);

    printf("clCreateKernel ...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clCreateKernel, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Set the Argument values

    ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&cmDevData);
	ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&cmCovx);
	//ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmCovx_inv);
	ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmMeanDiff);
	ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_mem), (void*)&cmPdf);
	ciErr1 |= clSetKernelArg(ckKernel, 4, sizeof(cl_mem), (void*)&cmMu);
	ciErr1 |= clSetKernelArg(ckKernel, 5, sizeof(cl_mem), (void*)&cmPhi);
	ciErr1 |= clSetKernelArg(ckKernel, 6, sizeof(cl_mem), (void*)&cmPdf_w);
	ciErr1 |= clSetKernelArg(ckKernel, 7, sizeof(cl_mem), (void*)&cmW);
    ciErr1 |= clSetKernelArg(ckKernel, 8, sizeof(cl_int), (void*)&N);
    ciErr1 |= clSetKernelArg(ckKernel, 9, sizeof(cl_int), (void*)&dim);
	ciErr1 |= clSetKernelArg(ckKernel, 10, sizeof(cl_int), (void*)&K);

    printf("clSetKernelArg 0 - 8...\n\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clSetKernelArg, Line %u in file %s !!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // --------------------------------------------------------
    // Start Core sequence... copy input data to GPU, compute, copy results back

    // Asynchronous write of data to GPU device
    ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, cmDevData, CL_FALSE, 0, sizeof(cl_float) *N*dim, data, 0, NULL, NULL);
	ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmMu, CL_FALSE, 0, sizeof(cl_float) *K*dim, mu, 0, NULL, NULL);
	ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmPhi, CL_FALSE, 0, sizeof(cl_float) *K, phi, 0, NULL, NULL);
	ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmCovx, CL_FALSE, 0, sizeof(cl_float) *K*dim*dim, covx, 0, NULL, NULL);
    printf("clEnqueueWriteBuffer (data and center)...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Launch kernel
	ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);

    printf("clEnqueueNDRangeKernel (cluster)...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Synchronous/blocking read of results, and check accumulated errors

	//ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmCovx_inv, CL_TRUE, 0, sizeof(cl_float) * dim*dim, covx_inv, 0, NULL, NULL);
	ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmMeanDiff, CL_TRUE, 0, sizeof(cl_float) * N*dim, meanDiff, 0, NULL, NULL);
	ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmPdf, CL_TRUE, 0, sizeof(cl_float) * N * K, pdf, 0, NULL, NULL);
	ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmPdf_w, CL_TRUE, 0, sizeof(cl_float) * N * K, pdf_w, 0, NULL, NULL);
	ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmW, CL_TRUE, 0, sizeof(cl_float) * N * K, W, 0, NULL, NULL);

    printf("clEnqueueReadBuffer (C)...\n\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clEnqueueReadBuffer, Line %u in file %s!!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    //--------------------------------------------------------

    // Compute and compare results for golden-host and report errors and pass/fail

	//for(i = 0; i<4; i++){
	//		printf("%f\t",(((float *)covx_inv)[i]));
	//}

	//for(i = 0; i<N*dim; i++){
	//	printf("%f\t",(((float *)meanDiff)[i]));
	//	if(i%dim == 1) printf("\n");
	//}

	fp = fopen("result.txt","w");
	for(i = 0; i<N * K; i++){
		printf("%f\t",(((float *)W)[i]));
		if((i+1)%K == 0) printf("\n");
		fprintf(fp,"%f\t",(((float *)W)[i]));
		if((i+1)%K == 0) fprintf(fp,"\n");
	}
	fclose(fp);

	Cleanup (argc, argv, 0);

    printf("end\n");
	system("pause");
}

void Cleanup (int argc, char **argv, int iExitCode)
{

	if(ckKernel)clReleaseKernel(ckKernel);
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);

	if(cmDevData)clReleaseMemObject(cmDevData);
	if(cmCovx) clReleaseMemObject(cmCovx);
	if(cmCovx_inv) clReleaseMemObject(cmCovx_inv);
	if(cmMeanDiff) clReleaseMemObject(cmMeanDiff);
	if(cmPdf) clReleaseMemObject(cmPdf);
	if(cmMu) clReleaseMemObject(cmMu);
	if(cmPdf_w) clReleaseMemObject(cmPdf_w);
	if(cmW) clReleaseMemObject(cmW);
    // Free host memory
    free(data);
	free(covx);
	free(covx_inv);
	free(meanDiff);
	free(mu);
	free(pdf);
	free(pdf_w);
	free(W);
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
