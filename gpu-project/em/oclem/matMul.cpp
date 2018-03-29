#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

//const char* cSourceFile = "update_center.cl";
#define matrix_dim 1

// Host buffers for demo
// *********************************************************************
void *data;      // Host buffers for OpenCL test
void *covx, *covx_inv;

// demo config vars
size_t N = 1000;	//number of nodes
//size_t	K = 4;		//number of clusters
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
    szGlobalWorkSize[0] = N;  // rounded up to the nearest multiple of the LocalWorkSize
	printf("...\n");
	printf("local:%d\t global:%d\n",szLocalWorkSize[0], szGlobalWorkSize[0]);
	//printf("global work size: %d; local size:%d; ", szGlobalWorkSize[0], szLocalWorkSize[0]);

    // Allocate and initialize host arrays
    data = (void *)malloc(sizeof(cl_float) * N *dim);
	covx = (void *)malloc(sizeof(cl_float) * dim *dim);
	covx_inv = (void *)malloc(sizeof(cl_float) * dim *dim);

	if (data == NULL || covx == NULL || covx_inv == NULL){
		printf("malloc failed\n");
		exit(-1);
	}
    fp = fopen("data.txt","r");
	for(i = 0; i<N*dim; i++){
		fscanf(fp, "%f",&(((float *)data)[i]));
		//printf("%f\t", (((float *)data)[i]));
		//if(i%2==1) printf("\n");
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
	cmCovx = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float) * dim * dim, NULL, &ciErr2);
	ciErr1 |= ciErr2;
	cmCovx_inv = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float) * dim * dim, NULL, &ciErr2);
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
	ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmCovx_inv);
    ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_int), (void*)&N);
    ciErr1 |= clSetKernelArg(ckKernel, 4, sizeof(cl_int), (void*)&dim);


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
        printf("Error %d in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n",ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

    // Synchronous/blocking read of results, and check accumulated errors

	ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmCovx, CL_TRUE, 0, sizeof(cl_float) * dim*dim, covx, 0, NULL, NULL);
	//ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmCovx_inv, CL_TRUE, 0, sizeof(cl_float) * dim*dim, covx_inv, 0, NULL, NULL);
    printf("clEnqueueReadBuffer (C)...\n\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error %d in clEnqueueReadBuffer, Line %u in file %s!!!\n\n", ciErr1, __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    //--------------------------------------------------------
	printf("printing\n");
    // Compute and compare results for golden-host and report errors and pass/fail
	printf("%f\n", ((float *)covx)[0]);
	for(i=0; i<4; i++){
		printf("%d: %f\n", i, ((float *)covx)[i]);
	}
	
	//fclose(fp);
	Cleanup (argc, argv, 0);

    printf("end\n");
	return 0;
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
    // Free host memory
    free(data);
	free(covx);
	free(covx_inv);
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
