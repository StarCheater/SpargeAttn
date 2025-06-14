import sys
import os
import subprocess
from pathlib import Path
from packaging.version import parse, Version
from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# Constants
SUPPORTED_ARCHS = {"8.0", "8.6", "8.9", "9.0"}
MIN_CUDA_VERSION = Version("12.0")
PROJECT_NAME = "spas_sage_attn"
VERSION = "0.1.0"

def validate_cuda_installation():
    """Validate CUDA Toolkit installation and version"""
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME not found. CUDA 12.0+ required")
    
    nvcc_path = Path(CUDA_HOME) / "bin" / "nvcc"
    if not nvcc_path.exists():
        raise RuntimeError(f"nvcc not found at {nvcc_path}")
    
    nvcc_version = subprocess.check_output([str(nvcc_path), "--version"], text=True)
    version_str = nvcc_version.split()[-2].split(",")[0]
    cuda_version = parse(version_str)
    
    if cuda_version < MIN_CUDA_VERSION:
        raise RuntimeError(f"Requires CUDA {MIN_CUDA_VERSION}+, found {cuda_version}")

def get_arch_flags():
    """Generate architecture-specific compilation flags"""
    arch_flags = []
    compute_capabilities = os.environ.get("TORCH_CUDA_ARCH_LIST", "").replace(" ", ";").split(";")
    
    for arch in compute_capabilities:
        if not arch:
            continue
        if arch in SUPPORTED_ARCHS:
            num = arch.replace(".", "")
            arch_flags.extend(["-gencode", f"arch=compute_{num},code=sm_{num}"])
        else:
            print(f"Warning: Unsupported architecture {arch} will be ignored")
    
    return arch_flags

def collect_sources(src_dirs):
    """Collect CUDA source files from specified directories"""
    sources = []
    for src_dir in src_dirs:
        base_path = Path(src_dir)
        sources.extend([
            str(p) for p in base_path.rglob("*.cu") 
            if p.is_file() and not p.name.startswith(".")
        ])
    return sources

# Main setup configuration
validate_cuda_installation()

extensions = [
    CUDAExtension(
        name=f"{PROJECT_NAME}._qattn",
        sources=[
            "csrc/qattn/pybind.cpp",
            *collect_sources([
                "csrc/qattn",
                "csrc/qattn/instantiations_sm80",
                "csrc/qattn/instantiations_sm89"
            ])
        ],
        extra_compile_args={
            "cxx": ["-O3", "-fopenmp", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "--use_fast_math",
                "--threads=8",
                *get_arch_flags()
            ]
        }
    ),
    CUDAExtension(
        name=f"{PROJECT_NAME}._fused",
        sources=collect_sources(["csrc/fused"]),
        extra_compile_args={
            "cxx": ["-O3", "-fopenmp", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "--use_fast_math",
                *get_arch_flags()
            ]
        }
    )
]

# Platform-specific configurations
if sys.platform == "win32":
    for ext in extensions:
        ext.extra_compile_args["nvcc"].extend([
            "-Xcompiler=/MD",
            "-Xcompiler=/wd4819"  # Suppress locale warnings
        ])
    extras = {"triton": ["triton-windows>=3.3.1"]}
else:
    extras = {"triton": ["triton>=3.2.0"]}

setup(
    name=PROJECT_NAME,
    version=VERSION,
    author="SpargeAttn Team",
    packages=find_packages(),
    description="High-performance Sparse Attention with CUDA",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/thu-ml/SpargeAttn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows"
    ],
    python_requires=">=3.9",
    install_requires=["torch>=2.3.0"],
    extras_require=extras,
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False
)
