"""Fixtures for every test file in tests/. pytest loads this file by itself."""

import pytest
import torch


@pytest.fixture(autouse=True)
def cuda_state():
    """Around every test: true float32 for the reference, and GPU errors reported in the test that caused them."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False   # with TF32, A @ B keeps ~10 mantissa bits
    yield
    torch.cuda.synchronize()                        # a kernel runs async: wait, so its error fails THIS test
    torch.backends.cuda.matmul.allow_tf32 = old
