CXXFLAGS=-Wall -O3 `pkg-config --cflags OpenCL`
LDFLAGS=`pkg-config --libs OpenCL`

all: himeno

himeno: himeno.cc
	g++ $(CXXFLAGS) himeno.cc -o himeno -g $(LDFLAGS)

clean:
	rm -rf himeno
