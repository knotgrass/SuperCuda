#include <iostream>

void vector_sum(const float *vector, const size_t n, double &result) {
    result = 0.0;
    for (size_t i = 0; i < n; i++) {
        result += vector[i];
    }
}

int main(int argc, char **argv) {
    const size_t n = 100'000'000;
    float *vector = new float[n];
    double result;

    // init vector
    for (size_t i = 0; i < n; i++) {
        vector[i] = 1.0;
    }

    // compute sum
    vector_sum(vector, n, result);
    std::cout << "result = " << result << std::endl;

    // free memory
    delete[] vector;

    return 0;
}
