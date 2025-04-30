# SLEEF expf Function Test

This example demonstrates the use of the exponential function (expf) from the SLEEF library.

## Building the Example

To build the example, simply run:

```bash
make
```

This will compile the program using the SLEEF headers from the build directory.

## Running the Test

After building, run the test with:

```bash
./expf_test
```

The program will:
1. Compare the results of standard expf and SLEEF's implementation for various inputs
2. Perform a simple performance test with 10 million function calls

## Notes

- The program links with the standard math library (-lm) for the comparison
- SLEEF's implementation is accessed through the Sleef_expf1_u10 function
- For more advanced usage, you can also try other variants like Sleef_expf1_u35 for different precision 