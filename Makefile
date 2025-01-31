#CUDA_HEADER_DIR=/usr/local/cuda-10.2/include
CUDA_HEADER_DIR=/usr/local/cuda-12.5/include
CXX_FLAGS=-Wall -D__CL_ENABLE_EXCEPTIONS -DCL_HPP_TARGET_OPENCL_VERSION=300 #-DDEBUG

all: himeno

himeno: himeno.cc
	g++ $(CXX_FLAGS) -I$(CUDA_HEADER_DIR) himeno.cc -o himeno -g -lOpenCL

clean:
	rm -rf himeno
