#include <CL/cl.h>
#include "FreeImage.h"
#include "time.h"

#include <time.h>
#include <iostream>
#include <sstream>
#include <fstream>


using namespace std;

cl_int binSize;					//the size of bin, in this program, make it 256
cl_int groupSize;				//size of work group
cl_int subHistogramNum;			//the number of total sub histograms
cl_uchar *data;					//contain the data of a image and write it to device
cl_int width;					//width of the image
cl_int height;					//height of the image
cl_float *cpuHistogram;			//the histogram computed by sequential cpu
cl_float *subDeviceHistogram;   //sub histogram computed by device
cl_float *deviceHistogram;      //the final histogram combining subDeviceHistogram
cl_mem deviceImageBuffer;          //the device memory to store the image data
cl_mem subDeviceHistogramBuffer;   //a buffer to store the sub histogram computed in device

const char* filename = "../wuzhen.jpg";

cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id PlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &PlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        cerr << "Failed to find any OpenCL platforms." << endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)PlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        cout << "Could not create GPU context, trying CPU..." << endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            cerr << "Failed to create an OpenCL GPU or CPU context." << endl;
            return NULL;
        }
    }

	cout << "Create context successful." << endl;
    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        delete [] devices;
        cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;

	cout << "Create commandqueue successful." << endl;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    ifstream kernelFile(fileName, ios::in);
    if (!kernelFile.is_open())
    {
        cerr << "Failed to open file for reading: " << fileName << endl;
        return NULL;
    }

    ostringstream oss;
    oss << kernelFile.rdbuf();

    string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        cerr << "Failed to create CL program from source." << endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[10240];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        cerr << "Error in kernel: " << endl;
        cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

	cout << "Create program successful." << endl;
    return program;
}

int main(int argc, char* argv[]){
	cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_int errNum;
	int i, j;

	//initialize the freeimage and load the jpg
	FIBITMAP *bitmap;
	unsigned char *imageData;
	FREE_IMAGE_FORMAT imagetype = FIF_UNKNOWN;
	int width;
	int height;
	FreeImage_Initialise(TRUE);

	//get the image type of the image
	imagetype = FreeImage_GetFileType(filename,0);
	if(imagetype == FIF_UNKNOWN)
		imagetype = FreeImage_GetFIFFromFilename(filename);
	if((imagetype != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(imagetype)) {
		//load the image to bitmap
		bitmap = FreeImage_Load(imagetype,filename,0);
		if(!bitmap){
			cout << "Image could not be loaded." <<endl;
			return 1;
		}
	}
	else{
		cout << "Image Format could not be recognized" << endl;
		return 1;
	}

	//coverte the bitmap to grey
	bitmap = FreeImage_ConvertTo8Bits(bitmap);

	//get the image data of image and the widht, height
	imageData = (unsigned char *)FreeImage_GetBits(bitmap);
	width = FreeImage_GetWidth(bitmap);
	height = FreeImage_GetHeight(bitmap);

	//set the binsize and workgroup size
	//the size of it need to be computed local memory in device
	binSize = 256;
	groupSize = 32;
	subHistogramNum = (width *height)/(binSize * groupSize);

	// allocate the memory for image data
	data = (cl_uchar*)malloc(width * height * sizeof(cl_uchar));
	if(!data){
		cerr << "Fail to allocate memory." <<endl;
		return 1;
	}
	memset(data, 0, width * height * sizeof(cl_uchar));
	
	//get the data
	for(i = 0; i < width * height; i++)
		data[i] =  (cl_uchar)imageData[i]; 

	//allocate the memory for histogram computed by cpu sequential
	cpuHistogram = (cl_float*)malloc(binSize * sizeof(cl_float));
	if(!cpuHistogram){
		cerr << "Fail to allocate memory." <<endl;
		return 1;
	}
	memset(cpuHistogram, 0, binSize * sizeof(cl_float));

	//allocate the memory for sub histogram computed in device
	subDeviceHistogram = (cl_float*)malloc(binSize * subHistogramNum * sizeof(cl_float));
	if(!subDeviceHistogram){
		cerr << "Fail to allocate memory." <<endl;
		return 1;
	}
	memset(subDeviceHistogram, 0, binSize * subHistogramNum * sizeof(cl_float));

	//allocate the memory for histogram computed in device
	deviceHistogram = (cl_float*)malloc(binSize * sizeof(cl_float));
	if(!deviceHistogram){
		cerr << "Fail to allocate memory." <<endl;
		return 1;
	}
	memset(deviceHistogram, 0, binSize * sizeof(cl_float));

	// Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL){
        cerr << "Failed to create OpenCL context." << endl;
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL){
		cerr << "Failed to create OpenCL commandQueue." << endl;
        return 1;
    }

    // Create OpenCL program from HelloWorld.cl kernel source
    program = CreateProgram(context, device, "histogram.cl");
    if (program == NULL){
		cerr << "Failed to create OpenCL program." << endl;
        return 1;
    }

    // Create OpenCL kernel
    kernel = clCreateKernel(program, "histogram", NULL);
    if (kernel == NULL){
        cerr << "Failed to create kernel." << endl;
        return 1;
    }

	//Create the buffer in device
	deviceImageBuffer = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
		sizeof(cl_uchar) * width  * height, data,  0);

	subDeviceHistogramBuffer = clCreateBuffer( context,  CL_MEM_READ_WRITE, 
		sizeof(cl_float) * binSize * subHistogramNum, NULL, 0);

	//the the arg in kernel
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&deviceImageBuffer); 
	errNum = clSetKernelArg(kernel, 1, sizeof(int), &width);
	errNum = clSetKernelArg(kernel, 2, sizeof(int), &height);
	errNum = clSetKernelArg(kernel, 3, groupSize * binSize * sizeof(cl_float), NULL);
	errNum = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&subDeviceHistogramBuffer); 
	if (errNum != CL_SUCCESS){
        cerr << "Error setting kernel arguments." << endl;
        return 1;
    }

	//globalThreads is the total number of workgroups
	//localThreads is the number of one workgroup
	size_t globalThreads = (width * height) / binSize;
	size_t localThreads = groupSize;

	//this is a quite necessary step to get the size of device memory
	//the work item's size should not exceed it
	/*
	cl_ulong local_mem_size;
	clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
	cout<<local_mem_size<<endl;
	*/
	/*
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 *           compute the histogram in device             *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 */
	//get the time of computing histogram in device
	clock_t deviceStart, deviceEnd;

	deviceStart = clock();
	//enqueue the command queue and compute the sub histogram in device
	clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, 
		&globalThreads, &localThreads, 0, NULL, NULL);
	if (errNum != CL_SUCCESS)
    {
        cerr << "Error queuing kernel for execution." << endl;
        return 1;
    }

	//read the sub histogram from device back to host
	errNum = clEnqueueReadBuffer(commandQueue, subDeviceHistogramBuffer, CL_TRUE, 0,
		subHistogramNum * binSize * sizeof(cl_float), subDeviceHistogram, 0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        cerr << "Error reading result buffer." << endl;
        return 1;
    }

	//combine the result return by different work groups
	for(i = 0; i < subHistogramNum; ++i){
		for( j = 0; j < binSize; ++j){
			deviceHistogram[j] += subDeviceHistogram[i * binSize + j];
		}
	}

	//enhance the image
	float temp;
	for(i = 1; i < 256; i++){
		temp = deviceHistogram[i] + deviceHistogram[i - 1];
		deviceHistogram[i] = (temp < (256 - 1))? temp : (256 - 1);
	}

	//set the image data according to the enhanced histogram
	for(i = 0; i < height; i++)
		for(j = 0; j < width; j++)
			imageData[i*width + j] = (unsigned char)deviceHistogram[(int)imageData[i*width + j]];

	deviceEnd = clock();
	cout << "The computing time in device is "<< (double)(deviceEnd-deviceStart)/CLOCKS_PER_SEC << "second."<<endl;

	//save the image
	FreeImage_Save(imagetype, bitmap, "../opencl_wuzhen.jpg", 0);

	/*
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 *           compute the histogram in cpu                *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 */
	//get the time of computing histogram in cpu
	clock_t cpuStart, cpuEnd;

	cpuStart = clock();
	//compute the histogram in cpu sequential
	for(i = 0; i < height; i++)
		for(j = 0; j < width; j++)
			cpuHistogram[(int)data[i * width + j]] += (float)256/(width * height);

	for(i = 1; i < 256; i++){
		temp = cpuHistogram[i] + cpuHistogram[i - 1];
		cpuHistogram[i] = (temp < (256 - 1))? temp: (256 - 1);
	}

	for(i = 0; i < height; i++)
		for(j = 0; j < width; j++)
			imageData[i*width + j] = (unsigned char)cpuHistogram[(int)data[i*width + j]];

	cpuEnd = clock();
	cout << "The computing time in cpu is "<< (double)(cpuEnd-cpuStart)/CLOCKS_PER_SEC << "second."<<endl;

	FreeImage_Save(imagetype, bitmap, "../cpu_wuzhen.jpg", 0);

	//free the memory used by program, release the resources used by opencl
	free(data);
	free(cpuHistogram);
	free(subDeviceHistogram);
	free(deviceHistogram);
	clReleaseProgram(program);
	clReleaseCommandQueue(commandQueue);
	clReleaseContext(context);
	return 0;
	}

