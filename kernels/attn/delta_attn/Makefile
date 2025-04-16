# GPU Selection: 4090, A100, H100
GPU_TARGET=4090

# Compiler
NVCC?=nvcc

# Common flags
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE -Xcompiler=-fPIC --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -I../../../include -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_4090 -arch=sm_80 -lcuda -lcudadevrt -lcudart_static

TARGET=deltanet
SRC=delta_attn_chunk_debug.cu

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)
