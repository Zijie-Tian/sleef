CXX = clang++
CXXFLAGS = -Wall -O3 -march=native -std=c++14
INCLUDES = -I../../build/include -I../../install/include
LDFLAGS = -L../../build/lib -L../../install/lib -lsleef -ltlfloat -lm -lbenchmark -lpthread

TARGET = expf_benchmark

all: $(TARGET)

$(TARGET): expf_benchmark.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean 