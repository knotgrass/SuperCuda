#define SIZE 100'000'000

void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

int main(int argc, char **argv){
    float *a, *b, *out;

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * SIZE);
    b   = (float*)malloc(sizeof(float) * SIZE);
    out = (float*)malloc(sizeof(float) * SIZE);

    // Initialize array
    for(int i = 0; i < SIZE; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // call the main function
    vector_add(out, a, b, SIZE);

    return 0;
}
