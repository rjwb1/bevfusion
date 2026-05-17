import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


def _detect_cuda_version():
    """Return (major, minor) of the nvcc that will actually be used, or None."""
    if CUDA_HOME is None:
        return None
    try:
        import subprocess
        out = subprocess.check_output(
            [os.path.join(CUDA_HOME, "bin", "nvcc"), "--version"]
        ).decode()
        # e.g. "Cuda compilation tools, release 11.8, V11.8.89"
        import re
        m = re.search(r"release (\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    return None


def _default_gencode_args():
    """Pick gencodes the installed nvcc actually supports.

    sm_86 needs CUDA >= 11.1, sm_89/sm_90 need CUDA >= 11.8.
    """
    archs = [(70, "sm_70"), (75, "sm_75"), (80, "sm_80")]
    cuda = _detect_cuda_version()
    if cuda is None or cuda >= (11, 1):
        archs.append((86, "sm_86"))
    if cuda is not None and cuda >= (11, 8):
        archs.append((89, "sm_89"))
        archs.append((90, "sm_90"))
    return [f"-gencode=arch=compute_{n},code={code}" for n, code in archs]


def make_cuda_ext(
    name, module, sources, sources_cuda=[], extra_args=[], extra_include_path=[]
):

    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        # Architecture list can be overridden by the standard TORCH_CUDA_ARCH_LIST
        # env var. Defaults below cover Volta (V100), Turing (T4/RTX 20xx),
        # Ampere (A100/RTX 30xx), Ada (RTX 40xx) and Hopper (H100); the gencode
        # set is filtered to what the installed nvcc actually supports.
        nvcc_args = [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        if os.getenv("TORCH_CUDA_ARCH_LIST") is None:
            nvcc_args += _default_gencode_args()
        extra_compile_args["nvcc"] = extra_args + nvcc_args
        sources += sources_cuda
    else:
        print("Compiling {} without CUDA".format(name))
        extension = CppExtension

    return extension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    setup(
        name="mmdet3d",
        packages=find_packages(),
        include_package_data=True,
        package_data={"mmdet3d.ops": ["*/*.so"]},
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        python_requires=">=3.8",
        license="Apache License 2.0",
        ext_modules=[
            make_cuda_ext(
                name="sparse_conv_ext",
                module="mmdet3d.ops.spconv",
                extra_include_path=[
                    # PyTorch 1.5 uses ninjia, which requires absolute path
                    # of included files, relative path will cause failure.
                    os.path.abspath(
                        os.path.join(*"mmdet3d.ops.spconv".split("."), "include/")
                    )
                ],
                sources=[
                    "src/all.cc",
                    "src/reordering.cc",
                    "src/reordering_cuda.cu",
                    "src/indice.cc",
                    "src/indice_cuda.cu",
                    "src/maxpool.cc",
                    "src/maxpool_cuda.cu",
                ],
                extra_args=["-w", "-std=c++17"],
            ),
            make_cuda_ext(
                name="bev_pool_ext",
                module="mmdet3d.ops.bev_pool",
                sources=[
                    "src/bev_pool.cpp",
                    "src/bev_pool_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="iou3d_cuda",
                module="mmdet3d.ops.iou3d",
                sources=[
                    "src/iou3d.cpp",
                    "src/iou3d_kernel.cu",
                ],
            ),
            make_cuda_ext(
                name="voxel_layer",
                module="mmdet3d.ops.voxel",
                sources=[
                    "src/voxelization.cpp",
                    "src/scatter_points_cpu.cpp",
                    "src/scatter_points_cuda.cu",
                    "src/voxelization_cpu.cpp",
                    "src/voxelization_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="roiaware_pool3d_ext",
                module="mmdet3d.ops.roiaware_pool3d",
                sources=[
                    "src/roiaware_pool3d.cpp",
                    "src/points_in_boxes_cpu.cpp",
                ],
                sources_cuda=[
                    "src/roiaware_pool3d_kernel.cu",
                    "src/points_in_boxes_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="ball_query_ext",
                module="mmdet3d.ops.ball_query",
                sources=["src/ball_query.cpp"],
                sources_cuda=["src/ball_query_cuda.cu"],
            ),
            make_cuda_ext(
                name="knn_ext",
                module="mmdet3d.ops.knn",
                sources=["src/knn.cpp"],
                sources_cuda=["src/knn_cuda.cu"],
            ),
            make_cuda_ext(
                name="assign_score_withk_ext",
                module="mmdet3d.ops.paconv",
                sources=["src/assign_score_withk.cpp"],
                sources_cuda=["src/assign_score_withk_cuda.cu"],
            ),
            make_cuda_ext(
                name="group_points_ext",
                module="mmdet3d.ops.group_points",
                sources=["src/group_points.cpp"],
                sources_cuda=["src/group_points_cuda.cu"],
            ),
            make_cuda_ext(
                name="interpolate_ext",
                module="mmdet3d.ops.interpolate",
                sources=["src/interpolate.cpp"],
                sources_cuda=["src/three_interpolate_cuda.cu", "src/three_nn_cuda.cu"],
            ),
            make_cuda_ext(
                name="furthest_point_sample_ext",
                module="mmdet3d.ops.furthest_point_sample",
                sources=["src/furthest_point_sample.cpp"],
                sources_cuda=["src/furthest_point_sample_cuda.cu"],
            ),
            make_cuda_ext(
                name="gather_points_ext",
                module="mmdet3d.ops.gather_points",
                sources=["src/gather_points.cpp"],
                sources_cuda=["src/gather_points_cuda.cu"],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )
