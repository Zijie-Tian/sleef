CC = clang++
CFLAGS = -Wall -O2
INCLUDES = -I../../build/include -I../../install/include
LDFLAGS = -L../../build/lib -L../../install/lib -lsleef -ltlfloat -lm 

TARGET = expf_test

all: $(TARGET)

$(TARGET): expf_test.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean 