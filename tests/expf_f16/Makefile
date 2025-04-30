CXX = clang++
CXXFLAGS = -Wall -O2 -std=c++14 -march=armv8.2-a+fp16+simd
INCLUDES = -I../../build/include -I../../install/include
LDFLAGS = -L../../build/lib -L../../install/lib -lsleef -ltlfloat -lm

TARGET = expf_f16_test

all: $(TARGET)

$(TARGET): expf_f16_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean 