# GPU Selection: 4090, A100, H100
GPU_TARGET=H100

# Compiler
NVCC=/usr/local/cuda-12.6/bin/nvcc

# Common flags
COMMON_FLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr \
            -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math \
            -I../../../include -forward-unknown-to-host-compiler -O3 \
            -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills \
            -std=c++20 -MD -MT -MF -x cu

# Libraries
LIBS=-lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas

# H100 specific flags
NVCCFLAGS=$(COMMON_FLAGS) -DKITTENS_HOPPER -arch=sm_90a $(LIBS)

TARGET=delta_attn
SRC=delta_attn.cu

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)
