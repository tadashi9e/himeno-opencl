CXX_FLAGS=-Wall -D__CL_ENABLE_EXCEPTIONS -DCL_HPP_TARGET_OPENCL_VERSION=220 #-DDEBUG

all: himeno

himeno: himeno.cc
	g++ $(CXX_FLAGS) himeno.cc -o himeno -g `pkg-config --libs --cflags OpenCL`

clean:
	rm -rf himeno
