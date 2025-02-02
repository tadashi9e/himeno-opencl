#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <numeric>
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#ifdef DEBUG
#define DEBUG_(msg) std::cout << msg << std::endl;
#define DEBUG_KERNEL(msg) command_queue.finish(); DEBUG_(msg);
#else  // DEBUG
#define DEBUG_(msg)
#define DEBUG_KERNEL(msg)
#endif  // DEBUG

static const char* KERNEL_SOURCE = "himeno_kernel.cl";

static cl::Platform platform;
static cl::Device device;
static cl::Context context;
static cl::CommandQueue command_queue;
static cl::Program program;
static cl::Kernel kernel_mat_set_init;
static cl::Kernel kernel_mat_set;
static cl::Kernel kernel_jacobi1;
static cl::Kernel kernel_sum;
static cl::Buffer dev_mat_p;
static cl::Buffer dev_mat_bnd;
static cl::Buffer dev_mat_wrk1;
static cl::Buffer dev_mat_wrk2;
static cl::Buffer dev_mat_a;
static cl::Buffer dev_mat_b;
static cl::Buffer dev_mat_c;
static cl::Buffer dev_mat_gosa;
static cl::Buffer dev_mat_sum_output;
static cl_int mimax;
static cl_int mjmax;
static cl_int mkmax;
static cl_int limax = 16;
static cl_int ljmax = 16;
static cl_int lkmax = 16;
static int GROUP_SIZE = 512;
static cl_float omega = 0.8f;
static size_t sum_size;
static std::vector<cl_float> zeros;
static std::vector<cl_float> sum_output;

static const char* current_execution = "initial";

double
fflop(int mx,int my, int mz)
{
  return((double)(mz-2)*(double)(my-2)*(double)(mx-2)*34.0);
}

double
mflops(int nn,double cpu,double flop)
{
  return(flop/cpu*1.e-6*(double)nn);
}

void
set_param(int is[],char *size)
{
  if(!strcmp(size,"XS") || !strcmp(size,"xs")){
    is[0]= 32;
    is[1]= 32;
    is[2]= 64;
    return;
  }
  if(!strcmp(size,"S") || !strcmp(size,"s")){
    is[0]= 64;
    is[1]= 64;
    is[2]= 128;
    return;
  }
  if(!strcmp(size,"M") || !strcmp(size,"m")){
    is[0]= 128;
    is[1]= 128;
    is[2]= 256;
    return;
  }
  if(!strcmp(size,"L") || !strcmp(size,"l")){
    is[0]= 256;
    is[1]= 256;
    is[2]= 512;
    return;
  }
  if(!strcmp(size,"XL") || !strcmp(size,"xl")){
    is[0]= 512;
    is[1]= 512;
    is[2]= 1024;
    return;
  } else {
    printf("Invalid input character !!\n");
    exit(6);
  }
}

cl::Buffer newMat(size_t mnums, size_t mrows, size_t mcols, size_t mdeps,
                  const char* name) {
  try {
    cl_int err;
    const size_t sz = sizeof(cl_float) * mnums * mrows * mcols * mdeps;
    std::cout << "newMat: " << sz << "=" << sizeof(cl_float)
              << "*" << mnums << "*" << mrows << "*" << mcols << "*" << mdeps
              << ": " << name << std::endl;
    cl::Buffer buff(
        context, CL_MEM_READ_WRITE,
        sz, NULL, &err);
    if (err != 0) {
      throw cl::Error(err, "failed to allocate cl::Buffer");
    }
    return buff;
  } catch (...) {
    std::cerr << "failed to allocate cl::Buffer" << std::endl;
    throw;
  }
}

// ----------------------------------------------------------------------
// utility function
// ----------------------------------------------------------------------
static void report_cl_error(const cl::Error& err) {
  const char* s = "-";
  switch (err.err()) {
#define CASE_CL_CODE(code)\
    case code: s = #code; break
    CASE_CL_CODE(CL_DEVICE_NOT_FOUND);
    CASE_CL_CODE(CL_DEVICE_NOT_AVAILABLE);
    CASE_CL_CODE(CL_COMPILER_NOT_AVAILABLE);
    CASE_CL_CODE(CL_MEM_OBJECT_ALLOCATION_FAILURE);
    CASE_CL_CODE(CL_OUT_OF_RESOURCES);
    CASE_CL_CODE(CL_OUT_OF_HOST_MEMORY);
    CASE_CL_CODE(CL_PROFILING_INFO_NOT_AVAILABLE);
    CASE_CL_CODE(CL_MEM_COPY_OVERLAP);
    CASE_CL_CODE(CL_IMAGE_FORMAT_MISMATCH);
    CASE_CL_CODE(CL_IMAGE_FORMAT_NOT_SUPPORTED);
    CASE_CL_CODE(CL_BUILD_PROGRAM_FAILURE);
    CASE_CL_CODE(CL_MAP_FAILURE);
#ifdef CL_VERSION_1_1
    CASE_CL_CODE(CL_MISALIGNED_SUB_BUFFER_OFFSET);
    CASE_CL_CODE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
#endif
#ifdef CL_VERSION_1_2
    CASE_CL_CODE(CL_COMPILE_PROGRAM_FAILURE);
    CASE_CL_CODE(CL_LINKER_NOT_AVAILABLE);
    CASE_CL_CODE(CL_LINK_PROGRAM_FAILURE);
    CASE_CL_CODE(CL_DEVICE_PARTITION_FAILED);
    CASE_CL_CODE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
#endif
    CASE_CL_CODE(CL_INVALID_VALUE);
    CASE_CL_CODE(CL_INVALID_DEVICE_TYPE);
    CASE_CL_CODE(CL_INVALID_PLATFORM);
    CASE_CL_CODE(CL_INVALID_DEVICE);
    CASE_CL_CODE(CL_INVALID_CONTEXT);
    CASE_CL_CODE(CL_INVALID_QUEUE_PROPERTIES);
    CASE_CL_CODE(CL_INVALID_COMMAND_QUEUE);
    CASE_CL_CODE(CL_INVALID_HOST_PTR);
    CASE_CL_CODE(CL_INVALID_MEM_OBJECT);
    CASE_CL_CODE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
    CASE_CL_CODE(CL_INVALID_IMAGE_SIZE);
    CASE_CL_CODE(CL_INVALID_SAMPLER);
    CASE_CL_CODE(CL_INVALID_BINARY);
    CASE_CL_CODE(CL_INVALID_BUILD_OPTIONS);
    CASE_CL_CODE(CL_INVALID_PROGRAM);
    CASE_CL_CODE(CL_INVALID_PROGRAM_EXECUTABLE);
    CASE_CL_CODE(CL_INVALID_KERNEL_NAME);
    CASE_CL_CODE(CL_INVALID_KERNEL_DEFINITION);
    CASE_CL_CODE(CL_INVALID_KERNEL);
    CASE_CL_CODE(CL_INVALID_ARG_INDEX);
    CASE_CL_CODE(CL_INVALID_ARG_VALUE);
    CASE_CL_CODE(CL_INVALID_ARG_SIZE);
    CASE_CL_CODE(CL_INVALID_KERNEL_ARGS);
    CASE_CL_CODE(CL_INVALID_WORK_DIMENSION);
    CASE_CL_CODE(CL_INVALID_WORK_GROUP_SIZE);
    CASE_CL_CODE(CL_INVALID_WORK_ITEM_SIZE);
    CASE_CL_CODE(CL_INVALID_GLOBAL_OFFSET);
    CASE_CL_CODE(CL_INVALID_EVENT_WAIT_LIST);
    CASE_CL_CODE(CL_INVALID_EVENT);
    CASE_CL_CODE(CL_INVALID_OPERATION);
    CASE_CL_CODE(CL_INVALID_GL_OBJECT);
    CASE_CL_CODE(CL_INVALID_BUFFER_SIZE);
    CASE_CL_CODE(CL_INVALID_MIP_LEVEL);
    CASE_CL_CODE(CL_INVALID_GLOBAL_WORK_SIZE);
#ifdef CL_VERSION_1_1
    CASE_CL_CODE(CL_INVALID_PROPERTY);
#endif
#ifdef CL_VERSION_1_2
    CASE_CL_CODE(CL_INVALID_IMAGE_DESCRIPTOR);
    CASE_CL_CODE(CL_INVALID_COMPILER_OPTIONS);
    CASE_CL_CODE(CL_INVALID_LINKER_OPTIONS);
    CASE_CL_CODE(CL_INVALID_DEVICE_PARTITION_COUNT);
#endif
#ifdef CL_VERSION_2_0
    CASE_CL_CODE(CL_INVALID_PIPE_SIZE);
    CASE_CL_CODE(CL_INVALID_DEVICE_QUEUE);
#endif
#ifdef CL_VERSION_2_2
    CASE_CL_CODE(CL_INVALID_SPEC_ID);
    CASE_CL_CODE(CL_MAX_SIZE_RESTRICTION_EXCEEDED);
#endif
    CASE_CL_CODE(CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR);
  }
  std::cerr << "caught exception: " << err.what() <<
    ": " << s << "(" << err.err() << ")" << std::endl;
}

static std::string loadProgramSource(const char *filename) {
  std::ifstream ifs(filename);
  const std::string content((std::istreambuf_iterator<char>(ifs)),
                            (std::istreambuf_iterator<char>()));
  return content;
}

static void mat_set_init(cl::Buffer* Mat) {
  DEBUG_("mat_set_init(Mat)");
  try {
    kernel_mat_set_init.setArg(0, *Mat);
    command_queue.enqueueNDRangeKernel(
        kernel_mat_set_init,
        cl::NullRange,
        cl::NDRange(mimax, mjmax, mkmax),
        cl::NDRange(limax, ljmax, lkmax));
    command_queue.finish();
  } catch (...) {
    std::cerr << "failed to mat_set_init" << std::endl;
    throw;
  }
}

static void mat_set(cl::Buffer* Mat, cl_int l, cl_float z) {
  DEBUG_("mat_set(Mat," << l << "," << z << ")");
  try {
    kernel_mat_set.setArg(0, *Mat);
    kernel_mat_set.setArg(1, sizeof(cl_int), &l);
    kernel_mat_set.setArg(2, sizeof(cl_float), &z);
    command_queue.enqueueNDRangeKernel(
        kernel_mat_set,
        cl::NullRange,
        cl::NDRange(mimax, mjmax, mkmax),
        cl::NDRange(limax, ljmax, lkmax));
    command_queue.finish();
  } catch (...) {
    std::cerr << "failed to mat_set" << std::endl;
    throw;
  }
}

static float jacobi(
    int nn,
    cl::Buffer* M1, cl::Buffer* M2, cl::Buffer* M3,
    cl::Buffer* M4, cl::Buffer* M5, cl::Buffer* M6, cl::Buffer* M7) {
  DEBUG_("jacobi(" << nn << ")");
  cl_float gosa = 0.0f;
  DEBUG_("    setup jacobi1 kernel arguments");
  current_execution = "setup jacobi1 kernel arguments";
  kernel_jacobi1.setArg(0, dev_mat_a);
  kernel_jacobi1.setArg(1, dev_mat_b);
  kernel_jacobi1.setArg(2, dev_mat_c);
  kernel_jacobi1.setArg(3, dev_mat_p);
  kernel_jacobi1.setArg(4, dev_mat_bnd);
  kernel_jacobi1.setArg(5, dev_mat_wrk1);
  kernel_jacobi1.setArg(6, dev_mat_wrk2);
  kernel_jacobi1.setArg(7, dev_mat_gosa);
  kernel_jacobi1.setArg(8, sizeof(cl_float), &omega);
  DEBUG_("    setup sum kernel arguments");
  current_execution = "setup sum kernel arguments";
  kernel_sum.setArg(0, dev_mat_gosa);
  kernel_sum.setArg(1, dev_mat_sum_output);
  kernel_sum.setArg(2, sizeof(cl_float) * sum_size, NULL);
  for(int n=0 ; n<nn ; n++) {
    DEBUG_("trial " << n << " / " << nn);
    DEBUG_("    executing jacobi1 kernel");
    current_execution = "enqueueNDRangeKernel(jacobi1)";
    command_queue.enqueueNDRangeKernel(
        kernel_jacobi1,
        cl::NullRange,
        cl::NDRange(mimax, mjmax, mkmax),
        cl::NDRange(limax, ljmax, lkmax));
    command_queue.finish();
    DEBUG_KERNEL("    copy p := wrk2");
    current_execution = "enqueueCopyBuffer(from wrk2 to p)";
    command_queue.enqueueCopyBuffer(
        dev_mat_wrk2, dev_mat_p,
        0, 0,
        sizeof(cl_float) * mimax * mjmax * mkmax);
    command_queue.finish();
    DEBUG_KERNEL("    sum of gosa");
    current_execution = "enqueueWriteBuffer(sum_output as 0)";
    command_queue.enqueueWriteBuffer(
      dev_mat_sum_output, CL_TRUE, 0,
      sizeof(cl_float) * sum_size,
      &zeros.front());
    command_queue.finish();
    current_execution = "enqueueNDRangeKernel(sum)";
    DEBUG_KERNEL("        executing gosa sum kernel");
    command_queue.enqueueNDRangeKernel(
        kernel_sum,
        cl::NullRange,
        cl::NDRange(mimax * mjmax * mkmax),
        cl::NDRange(sum_size));
    command_queue.finish();
    DEBUG_KERNEL("        reading gosa");
    current_execution = "enqueueReadBuffer(sum_output)";
    command_queue.enqueueReadBuffer(dev_mat_sum_output,
                                    CL_TRUE, 0, sum_size, &sum_output[0]);
    command_queue.finish();
    DEBUG_KERNEL("        accumulating gosa");
    gosa = std::accumulate(
        sum_output.begin(), sum_output.end(), 0.0f);
    DEBUG_("    gosa=" << gosa);
  }
  current_execution = "finished jacobi1";
  return gosa;
}

double
second()
{

  struct timeval tm;
  double t ;

  static int base_sec = 0,base_usec = 0;

  gettimeofday(&tm, NULL);
  
  if(base_sec == 0 && base_usec == 0)
    {
      base_sec = tm.tv_sec;
      base_usec = tm.tv_usec;
      t = 0.0;
  } else {
    t = (double) (tm.tv_sec-base_sec) + 
      ((double) (tm.tv_usec-base_usec))/1.0e6 ;
  }

  return t ;
}

static void usage(const char* argv0) {
  std::cout << "usage: " << " [--device N] [size]" << std::endl;
  std::cout << "    N    : OpenCL device index" << std::endl;
  std::cout << "    size : Problem size (xs / s / m / l / xl)" << std::endl;
  exit(1);
}
int
main(int argc, char* argv[]) {
  size_t device_index = 0;
  int msize[3];
  char   size[10] = {0};

  for (int arg = 1; arg < argc; ++arg) {
    if (std::string(argv[arg]) == "--device") {
      if (++arg >= argc) {
        usage(argv[0]);
      }
      device_index = atoi(argv[arg]);
    } else {
      strcpy(size, argv[arg]);
    }
  }
  if (size[0] == 0) {
    printf("For example: \n");
    printf(" Grid-size= XS (32x32x64)\n");
    printf("\t    S  (64x64x128)\n");
    printf("\t    M  (128x128x256)\n");
    printf("\t    L  (256x256x512)\n");
    printf("\t    XL (512x512x1024)\n\n");
    printf("Grid-size = ");
    scanf("%s",size);
    printf("\n");
  }
  set_param(msize,size);

  mimax= msize[0];
  mjmax= msize[1];
  mkmax= msize[2];
  const int imax= mimax-1;
  const int jmax= mjmax-1;
  const int kmax= mkmax-1;

  const float target = 60.0;

  printf("mimax = %d mjmax = %d mkmax = %d\n",mimax,mjmax,mkmax);
  printf("imax = %d jmax = %d kmax =%d\n",imax,jmax,kmax);

  try {
    current_execution = "cl::Platform::get()";
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    bool found_device = false;
    size_t dev_index = 0;
    for (cl::Platform& plat : platforms) {
      std::vector<cl::Device> devices;
      const std::string platvendor = plat.getInfo<CL_PLATFORM_VENDOR>();
      const std::string platname = plat.getInfo<CL_PLATFORM_NAME>();
      const std::string platver = plat.getInfo<CL_PLATFORM_VERSION>();
      std::cout << "platform: vendor[" << platvendor << "]"
        ",name[" << platname << "]"
        ",version[" << platver << "]" << std::endl;
      plat.getDevices(CL_DEVICE_TYPE_GPU, &devices);
      for (cl::Device& dev : devices) {
        const std::string devvendor = dev.getInfo<CL_DEVICE_VENDOR>();
        const std::string devname = dev.getInfo<CL_DEVICE_NAME>();
        const std::string devver = dev.getInfo<CL_DEVICE_VERSION>();
        std::cout << ((dev_index == device_index) ? '*' : ' ') <<
          "device[" << dev_index << "]: vendor[" << devvendor << "]"
          ",name[" << devname << "]"
          ",version[" << devver << "]" << std::endl;
        size_t global_mem_size;
        dev.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE,
                    &global_mem_size);
        std::cout << "        DEVICE_GLOBAL_MEM_SIZE="
                  << global_mem_size << std::endl;
        size_t local_mem_size;
        dev.getInfo(CL_DEVICE_LOCAL_MEM_SIZE,
                    &local_mem_size);
        std::cout << "        DEVICE_LOCAL_MEM_SIZE="
                  << local_mem_size << std::endl;
        int max_compute_units;
        dev.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS,
                    &max_compute_units);
        std::cout << "        DEVICE_MAX_COMPUTE_UNITS="
                  << max_compute_units << std::endl;
        size_t max_work_group_size;
        dev.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    &max_work_group_size);
        std::cout << "        DEVICE_MAX_WORK_GROUP_SIZE="
                  << max_work_group_size << std::endl;
        if (dev_index == device_index) {
          GROUP_SIZE = max_work_group_size;
          while (limax * ljmax * lkmax > GROUP_SIZE) {
            limax /= 2;
            if (limax * ljmax * lkmax > GROUP_SIZE) {
              ljmax /= 2;
            }
            if (limax * ljmax * lkmax > GROUP_SIZE) {
              lkmax /= 2;
            }
          }
          std::cout <<
            "limax = " << limax << " "
            "ljmax = " << ljmax << " "
            "lkmax = " << lkmax << std::endl;
          for (sum_size = 2;
               sum_size * sum_size < local_mem_size;
               sum_size *= sum_size);
          std::cout <<
            "sum_size = " << sum_size << std::endl;
          platform = plat;
          device = dev;
          found_device = true;
        }
        ++dev_index;
      }
    }
    if (!found_device) {
      std::cerr << "device[" << device_index << "] not found" << std::endl;
      exit(1);
    }
    const cl_platform_id platform_id = device.getInfo<CL_DEVICE_PLATFORM>()();
    cl_context_properties properties[3];
    properties[0] = CL_CONTEXT_PLATFORM;
    properties[1] = reinterpret_cast<cl_context_properties>(platform_id);
    properties[2] = 0;

    current_execution = "cl::Context()";
    context = cl::Context(device, properties);
    command_queue = cl::CommandQueue(context, device, 0);

    DEBUG_("creating cl::Buffers...");
    current_execution = "cl::Buffers()";
    dev_mat_p = newMat(1,mimax,mjmax,mkmax, "p");
    dev_mat_bnd = newMat(1,mimax,mjmax,mkmax, "bnd");
    dev_mat_wrk1 = newMat(1,mimax,mjmax,mkmax, "wrk1");
    dev_mat_wrk2 = newMat(1,mimax,mjmax,mkmax, "wrk2");
    dev_mat_a = newMat(4,mimax,mjmax,mkmax, "a");
    dev_mat_b = newMat(3,mimax,mjmax,mkmax, "b");
    dev_mat_c = newMat(3,mimax,mjmax,mkmax, "c");
    dev_mat_gosa = newMat(1,mimax,mjmax,mkmax, "gosa");
    cl_int err;
    dev_mat_sum_output = cl::Buffer(
        context, CL_MEM_READ_WRITE,
        sizeof(cl_float) * sum_size, NULL, &err);
    if (err != 0) {
      throw cl::Error(err, "failed to allocate cl::Buffer");
    }

    DEBUG_("loading kernel source...");
    current_execution = "loading kernel source";
    const std::string source_string = loadProgramSource(KERNEL_SOURCE);
    current_execution = "cl::Program()";
    program = cl::Program(context, source_string);
    try {
      current_execution = "cl::Program::build()";
      program.build();
    } catch (const cl::Error& err) {
      std::cout << "Build Status: "
                << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(
                    context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
      std::cout << "Build Options: "
                << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(
                    context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
      std::cout << "Build Log: "
                << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
                    context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
      throw err;
    }

    DEBUG_("setting up mat_set_init kernel...");
    current_execution = "cl::Kernel(mat_set_init)";
    kernel_mat_set_init = cl::Kernel(program, "mat_set_init");

    DEBUG_("setting up mat_set kernel...");
    current_execution = "cl::Kernel(mat_set)";
    kernel_mat_set = cl::Kernel(program, "mat_set");

    DEBUG_("setting up jacobi1 kernel...");
    current_execution = "cl::Kernel(jacobi1)";
    kernel_jacobi1 = cl::Kernel(program, "jacobi1");

    DEBUG_("setting up sum kernel...");
    current_execution = "cl::Kernel(sum)";
    kernel_sum = cl::Kernel(program, "sum");

    current_execution = "set up matrix";
    mat_set_init(&dev_mat_p);
    mat_set(&dev_mat_bnd,0,1.0);
    mat_set(&dev_mat_wrk1,0,0.0);
    mat_set(&dev_mat_wrk2,0,0.0);
    mat_set(&dev_mat_a,0,1.0);
    mat_set(&dev_mat_a,1,1.0);
    mat_set(&dev_mat_a,2,1.0);
    mat_set(&dev_mat_a,3,1.0/6.0);
    mat_set(&dev_mat_b,0,0.0);
    mat_set(&dev_mat_b,1,0.0);
    mat_set(&dev_mat_b,2,0.0);
    mat_set(&dev_mat_c,0,1.0);
    mat_set(&dev_mat_c,1,1.0);
    mat_set(&dev_mat_c,2,1.0);

    const size_t sz = sizeof(cl_float) * sum_size;
    zeros.resize(sizeof(cl_float) * sz);
    sum_output.resize(sizeof(cl_float) * sz);

    /*
     *    Start measuring
     */
    int nn;
    float gosa;
    double cpu0, cpu1, cpu;
    double flop;

    nn= 3;
    printf(" Start rehearsal measurement process.\n");
    printf(" Measure the performance in %d times.\n\n",nn);

    cpu0= second();
    gosa= jacobi(nn,&dev_mat_a,&dev_mat_b,&dev_mat_c,
                 &dev_mat_p,&dev_mat_bnd,
                 &dev_mat_wrk1,&dev_mat_wrk2);
    cpu1= second();
    cpu= cpu1 - cpu0;
    flop= fflop(imax,jmax,kmax);

    printf(" MFLOPS: %f time(s): %f %e\n\n",
           mflops(nn,cpu,flop),cpu,gosa);

    nn= (int)(target/(cpu/3.0));

    printf(" Now, start the actual measurement process.\n");
    printf(" The loop will be excuted in %d times\n",nn);
    printf(" This will take about one minute.\n");
    printf(" Wait for a while\n\n");

    cpu0 = second();
    gosa = jacobi(nn,&dev_mat_a,&dev_mat_b,&dev_mat_c,
                  &dev_mat_p,&dev_mat_bnd,
                  &dev_mat_wrk1,&dev_mat_wrk2);
    cpu1 = second();
    cpu = cpu1 - cpu0;

    printf(" Loop executed for %d times\n",nn);
    printf(" Gosa : %e \n",gosa);
    printf(" MFLOPS measured : %f\tcpu : %f\n",mflops(nn,cpu,flop),cpu);
    printf(" Score based on Pentium III 600MHz using Fortran 77: %f\n",
           mflops(nn,cpu,flop)/82);
  } catch (const cl::Error& err) {
    std::cerr << current_execution << ": ";
    report_cl_error(err);
  }
  return 0;
}
