# GPU Selection: 4090, H100  (e.g. `make GPU_TARGET=H100`)
GPU_TARGET?=4090

# Compiler
NVCC?=nvcc

# Common flags
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE -Xcompiler=-fPIC --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -I../../../include -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lineinfo

ifeq ($(GPU_TARGET),4090)
	SRC=4090.cu
	NVCCFLAGS+=-DKITTENS_4090 -arch=sm_80
else ifeq ($(GPU_TARGET),H100)
	SRC=h100.cu
	NVCCFLAGS+=-DKITTENS_HOPPER -arch=sm_90a
else
    $(error Unsupported GPU_TARGET '$(GPU_TARGET)', use 4090 or H100)
endif

TARGET=deltanet

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p printouts
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)
