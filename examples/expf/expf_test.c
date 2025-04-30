#include <stdio.h>
#include <math.h>
#include "sleef.h"

int main() {
    printf("Testing SLEEF expf functions\n");
    printf("============================\n\n");
    
    float test_values[] = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 10.0f};
    int num_values = sizeof(test_values) / sizeof(test_values[0]);
    
    printf("%-10s %-15s %-15s %-15s\n", "Input", "Standard expf", "SLEEF expf_u10", "Difference");
    printf("%-10s %-15s %-15s %-15s\n", "-----", "-------------", "-------------", "----------");
    
    for (int i = 0; i < num_values; i++) {
        float input = test_values[i];
        float std_result = expf(input);
        float sleef_result = Sleef_expf1_u10(input);
        float diff = fabsf(std_result - sleef_result);
        
        printf("%-10.2f %-15.8g %-15.8g %-15.8g\n", 
               input, std_result, sleef_result, diff);
    }
    
    printf("\n");
    
    // Test performance (simple demonstration)
    printf("Performance test - calculating exp(0.5) 10 million times\n");
    float result = 0.0f;
    float x = 0.5f;
    
    printf("Using standard library expf...\n");
    for (int i = 0; i < 10000000; i++) {
        result += expf(x);
        x += 0.000000001f; // Prevent optimization
    }
    printf("Done. Dummy result: %f\n\n", result);
    
    result = 0.0f;
    x = 0.5f;
    
    printf("Using SLEEF Sleef_expf1_u10...\n");
    for (int i = 0; i < 10000000; i++) {
        result += Sleef_expf1_u10(x);
        x += 0.000000001f; // Prevent optimization
    }
    printf("Done. Dummy result: %f\n", result);
    
    return 0;
} 