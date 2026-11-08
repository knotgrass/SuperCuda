#pragma once
// Stand-in for the registration.h that kernel-builder generates for Hub builds.
// torch_binding.cpp includes this file in the dev (JIT) build; the macros are the same.
#include <Python.h>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

// TORCH_LIBRARY with the name expanded first, so NAME can be a macro (TORCH_EXTENSION_NAME).
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

// Defines PyInit_<NAME>, so Python's import can load the .so. The module itself is empty:
// the ops are registered by the TORCH_LIBRARY static initializer when the .so is loaded.
#define REGISTER_EXTENSION(NAME)                                                                   \
    PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                                       \
        static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STRINGIFY(NAME), nullptr, 0, nullptr}; \
        return PyModule_Create(&module);                                                           \
    }
