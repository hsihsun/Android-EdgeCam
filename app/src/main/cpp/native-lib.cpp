#include <jni.h>
#include <android/log.h>
#define LOG_TAG "JNIpart"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/video.hpp>
#include "opencv2/core/utility.hpp"
#include <iostream>
#include <omp.h>
#include <sys/time.h>
#include <fstream>
#include "flow_functions.h"
#include "Utils.h"
#define CL_TARGET_OPENCL_VERSION 120
#define work_item_size 32

#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))

#include<CL/cl.h>
//#include<CL/cl2.hpp>
#include<CL/cl_platform.h>

#include <GLES2/gl2.h>
#include <EGL/egl.h>

#define TAG "NativeLib"

using namespace std;
using namespace cv;

cl_platform_id platformId = NULL;
cl_device_id deviceID = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_uint retNumDevices;
cl_uint retNumPlatforms;
cl_program program =NULL;

extern "C" void JNICALL Java_com_example_nightsight_MainActivity_initCL(JNIEnv *env, jobject instance, jstring openCLProgramText) {
    cl_int  error;

    cl_int ret = clGetPlatformIDs(1, &platformId, &retNumPlatforms);
    ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices);

    context = clCreateContext(NULL, 1, &deviceID, NULL, NULL,  &error);

    if(error != CL_SUCCESS) {
        LOGD("Can't create a valid OpenCL context");

    }else{
        LOGD("Create a valid OpenCL context");
    }

    char str_buffer[1024];

    // Get platform name
    error = clGetPlatformInfo(platformId, CL_PLATFORM_NAME, sizeof(str_buffer), &str_buffer, NULL);
    LOGD("CL_PLATFORM_NAME: %s", str_buffer);
    // Get platform vendor
    error = clGetPlatformInfo(platformId, CL_PLATFORM_VENDOR, sizeof(str_buffer), &str_buffer, NULL);
    LOGD("CL_PLATFORM_VENDOR: %s", str_buffer);
    // Get platform OpenCL version
    error = clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, sizeof(str_buffer), &str_buffer, NULL);
    LOGD("CL_PLATFORM_VERSION: %s", str_buffer);
    // Get platform OpenCL profile
    error = clGetPlatformInfo(platformId, CL_PLATFORM_PROFILE, sizeof(str_buffer), &str_buffer, NULL);
    LOGD("CL_PLATFORM_PROFILE: %s", str_buffer);
    // Get platform OpenCL supported extensions
    error = clGetPlatformInfo(platformId, CL_PLATFORM_EXTENSIONS, sizeof(str_buffer), &str_buffer, NULL);
    LOGD("CL_PLATFORM_EXTENSIONS: %s", str_buffer);

    error = clGetDeviceInfo(deviceID, CL_DEVICE_NAME, sizeof(str_buffer), &str_buffer, NULL);
    LOGD(" CL_DEVICE_NAME: %s", str_buffer);
    // Get device hardware version
    error = clGetDeviceInfo(deviceID, CL_DEVICE_VERSION, sizeof(str_buffer), &str_buffer, NULL);
    LOGD("  CL_DEVICE_VERSION: %s", str_buffer);
    // Get device software version
    error = clGetDeviceInfo(deviceID, CL_DRIVER_VERSION, sizeof(str_buffer), &str_buffer, NULL);
    LOGD(" CL_DRIVER_VERSION:  %s", str_buffer);
    // Get device OpenCL C version
    error = clGetDeviceInfo(deviceID, CL_DEVICE_OPENCL_C_VERSION, sizeof(str_buffer), &str_buffer, NULL);
    LOGD("CL_DEVICE_OPENCL_C_VERSION: %s", str_buffer);

    cl_bool imageSupport = CL_FALSE;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &imageSupport, NULL);
    if(imageSupport != CL_TRUE){
        LOGD("OpenCL device does not support images.");
    }else{
        LOGD("OpenCL device support images.");
    }

    commandQueue = clCreateCommandQueue(context, deviceID, 0, &error);

    const char* openCLProgramTextNative = env->GetStringUTFChars(openCLProgramText, 0);
    LOGD("OpenCL program text:\n%s", openCLProgramTextNative);

    program = clCreateProgramWithSource(context, 1, (const char**)&openCLProgramTextNative, NULL, &error);
    if ( error != CL_SUCCESS){
        LOGD("Failed to create program");
    }else{
        LOGD("Success to create program");
    }

    char *program_log;
    const char options[] = "";
    size_t log_size;

    error = clBuildProgram(program, 1, &deviceID, options, NULL, NULL);

    if (error != CL_SUCCESS)
    {
        size_t len;
        char buffer[8 * 1024];

        LOGD("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        LOGD("build program %s\n", buffer);
    }

    if ( error != CL_SUCCESS){
        LOGD("Failed to build program");
    }else{
        LOGD("Success to build program");
    }

}


extern "C" void JNICALL Java_com_example_nightsight_MainActivity_coreFiltering(JNIEnv *env, jobject instance,jlong grayAddr, jint mapFlag) {

    clock_t begin_total = clock();
    clock_t begin = clock();
    double totalTime = 0.0;
    // get Mat from raw address

    Mat &grayImg = *(Mat *) grayAddr;

  //  cv::cvtColor(grayImg, grayImg, cv::COLOR_RGBA2RGB);
    int width = grayImg.cols;
    int height = grayImg.rows;
    uchar *data = (uchar *)grayImg.data;

    LOGD("srcImg channels: %d", grayImg.channels());
    Mat dst(height, width,CV_8UC4);
    uchar *buffer = new uchar[width * height*4];

    cl_int  error;

    cl_image_format clImageFormat;
    clImageFormat.image_channel_order = CL_RGBA;
    clImageFormat.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc clImageDesc;
    clImageDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    clImageDesc.image_width = width;
    clImageDesc.image_height = height;
    clImageDesc.image_row_pitch = 0;
    clImageDesc.image_slice_pitch = 0;
    clImageDesc.num_mip_levels = 0;
    clImageDesc.num_samples = 0;
    clImageDesc.buffer = NULL;

    size_t origin[3] = { 0, 0, 0 };  //要读取的图像图像原点
    size_t region[3] = { (size_t) width, (size_t) height, 1};  //需要读取的图像区域，第三个参数是深度=1

    cl_mem srcImg = clCreateImage(context, CL_MEM_READ_WRITE, &clImageFormat, &clImageDesc, NULL, &error);
    if ( error != CL_SUCCESS){
        LOGD("Failed to create srcImg");
    }else{
        LOGD("Success to create srcImg");
    }
    error = clEnqueueWriteImage(commandQueue, srcImg , CL_TRUE, origin, region, 0, 0, data, 0, NULL, NULL);
    clFinish(commandQueue);
    if ( error != CL_SUCCESS){
        LOGD("Failed to write srcImg");
    }else{
        LOGD("Success to write  srcImg");
    }

    cl_mem dstImg = clCreateImage(context, CL_MEM_WRITE_ONLY, &clImageFormat, &clImageDesc, NULL, &error);

    if ( error != CL_SUCCESS){
        LOGD("Failed to create dstImg ");
    }else{
        LOGD("Success to create dstImg ");
    }


    size_t globalThreads[] = {static_cast<size_t>(ceil(width/16)*16), static_cast<size_t>(ceil(height/16)*16), };
    size_t localThreads[] = {16,16};

    cl_kernel kernel = clCreateKernel(program, "sobel_filter_color", &error);

    if ( error != CL_SUCCESS){
        LOGD("Failed to create kernel");
    }else{
        LOGD("Success to create kernel");
    }

    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &srcImg);
    error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dstImg);
    error |= clSetKernelArg(kernel, 2, sizeof(cl_int), &width);
    error |= clSetKernelArg(kernel, 3, sizeof(cl_int), &height);

    if ( error != CL_SUCCESS){
        LOGD("Error setting kernel arguments");
    }else{
        LOGD("Successfully setting kernel arguments");
    }
    begin = clock();
    error = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalThreads, localThreads, 0, NULL, NULL);
    clFinish(commandQueue);

    if ( error != CL_SUCCESS){
        LOGD("Error queueing kernel for execution");
    }else{
        LOGD("Successfully queueing kernel for execution");
    }

    totalTime = double(clock() - begin_total) / CLOCKS_PER_SEC;
    __android_log_print(ANDROID_LOG_INFO, TAG, "Execution time slice = %f seconds\n",
                        totalTime);

    error = clEnqueueReadImage(commandQueue, dstImg , CL_TRUE, origin, region, 0, 0, buffer, 0, NULL, NULL);

    if ( error != CL_SUCCESS){
        LOGD("Error to read dstImg srcImg");
    }else{
        LOGD("Successfully to read dstImg srcImg");
    }



    dst.data = (uchar*) buffer;
    dst.convertTo(dst, CV_8UC4);
   // cv::cvtColor(dst,dst, CV_RGBA2GRAY);
    grayImg = dst;

    LOGD("Size in JNI: %d X %d", dst.cols, dst.rows);

    clReleaseMemObject(srcImg);
    clReleaseMemObject(dstImg);

    totalTime = double(clock() - begin_total) / CLOCKS_PER_SEC;
    __android_log_print(ANDROID_LOG_INFO, TAG, "Total time slice = %f seconds\n",
                        totalTime);


}