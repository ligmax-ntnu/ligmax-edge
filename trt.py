"""Minimal TensorRT runner.

DeepStream's nvinfer would be the obvious choice, but reading its detection
metadata needs pyds, and neither pyds nor deepstream_python_apps is installed on
this box. Driving TensorRT directly is fewer moving parts and gives us the boxes
as plain numpy.
"""

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(_LOGGER, "")


def _check(err, what):
    if isinstance(err, tuple):
        err = err[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{what}: {err}")


class Engine:
    """One engine, one CUDA stream, preallocated device buffers.

    Buffers are allocated once and reused: at 14 fps on two cameras, per-frame
    cudaMalloc would be pure waste and would fragment.
    """

    def __init__(self, path):
        with open(path, "rb") as f:
            self.engine = trt.Runtime(_LOGGER).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {path}")
        self.ctx = self.engine.create_execution_context()

        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            nbytes = int(np.prod(shape)) * 4          # all fp32 at the boundary
            e = cudart.cudaMalloc(nbytes)
            _check(e, f"cudaMalloc {name}")
            rec = {"name": name, "shape": shape, "nbytes": nbytes, "dev": e[1]}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(rec)
            else:
                rec["host"] = np.empty(shape, dtype=np.float32)
                self.outputs.append(rec)
            self.ctx.set_tensor_address(name, int(rec["dev"]))

        e = cudart.cudaStreamCreate()
        _check(e, "cudaStreamCreate")
        self.stream = e[1]

        self.in_shape = self.inputs[0]["shape"]
        self.out_shape = self.outputs[0]["shape"]

    def infer(self, blob):
        """Run one batch. `blob` must match the input shape and be float32 C-contiguous."""
        inp = self.inputs[0]
        arr = np.ascontiguousarray(blob, dtype=np.float32)
        _check(cudart.cudaMemcpyAsync(
            inp["dev"], arr.ctypes.data, inp["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream), "h2d")
        if not self.ctx.execute_async_v3(self.stream):
            raise RuntimeError("execute_async_v3 failed")
        for o in self.outputs:
            _check(cudart.cudaMemcpyAsync(
                o["host"].ctypes.data, o["dev"], o["nbytes"],
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream), "d2h")
        _check(cudart.cudaStreamSynchronize(self.stream), "sync")
        return [o["host"] for o in self.outputs]

    def close(self):
        for rec in self.inputs + self.outputs:
            if rec.get("dev"):
                cudart.cudaFree(rec["dev"])
                rec["dev"] = None
        if getattr(self, "stream", None):
            cudart.cudaStreamDestroy(self.stream)
            self.stream = None
