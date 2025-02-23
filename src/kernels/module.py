from pathlib import Path

from torch.utils._triton import has_triton

if has_triton():
    try:
        import triton  # noqa

        HAS_TRITON = True
    except:  # noqa
        HAS_TRITON = False


# try:
#     import pycuda.autoprimaryctx
#     from pycuda.compiler import SourceModule

#     HAS_PYCUDA = True
#     _MODULES = {}
# except:  # noqa
HAS_PYCUDA = False
_MODULES = None


def get_source_module(cuda_file: str) -> callable:
    if not HAS_PYCUDA:
        raise ValueError("Should not be here without PyCuda!!!")

    with open(Path(__file__).resolve().parent / "cuda" / f"{cuda_file}.cu", "r") as f:
        src = "\n".join(f.readlines())

    def get_and_compile(src: str) -> callable:
        # x = SourceModule(src, options=["-std=c++14", "-O3"], no_extern_c=True)
        print(src)
        # return x.get_function("PYCUDA_APPLY_KERNEL")

    if cuda_file not in _MODULES:
        _MODULES[cuda_file] = get_and_compile(src)
    return _MODULES[cuda_file]


def implemented_as(cuda_func: callable = None, triton_func: callable = None):
    def decorator(func):
        if HAS_TRITON and triton_func is not None:
            func = triton_func
        elif HAS_PYCUDA and cuda_func is not None:
            func = cuda_func

        def inner(*args, **kwargs):
            return func(*args, **kwargs)

        return inner

    return decorator
