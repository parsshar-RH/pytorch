import torch
from torch._inductor.codegen.common import (
    DeviceOpOverrides,
    register_backend_for_device,
    register_device_op_overrides,
)
from torch._inductor.codegen.cpp import CppScheduling
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.virtualized import V


class OpenRegDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name: str) -> str:
        return (
            "\ndef get_raw_stream(_):\n"
            "    return 0\n"
        )

    def set_device(self, device_idx: int) -> str:
        return f"torch_openreg._C._set_device({device_idx})"

    def synchronize(self) -> str:
        return "pass"

    def device_guard(self, device_idx: int) -> str:
        return "torch._ops.contextlib.nullcontext()"

    def cpp_kernel_type(self) -> str:
        return "void*"


class OpenRegWrapperCodegen(PythonWrapperCodegen):

    @staticmethod
    def create(
        is_subgraph,
        subgraph_name,
        parent_wrapper,
        partition_signatures=None,
    ):
        if is_subgraph:
            from torch._inductor.codegen.wrapper import SubgraphPythonWrapperCodegen

            return SubgraphPythonWrapperCodegen(
                subgraph_name, parent_wrapper, partition_signatures
            )
        return OpenRegWrapperCodegen()

    def _generate_kernel_call_helper(
        self, kernel_name, call_args, *, device=None, **kwargs
    ):
        device = device or V.graph.get_current_device_or_throw()
        if device.type == "openreg":
            device = torch.device("cpu")
        super()._generate_kernel_call_helper(
            kernel_name, call_args, device=device, **kwargs
        )


register_backend_for_device(
    "openreg",
    CppScheduling,
    OpenRegWrapperCodegen,
)
register_device_op_overrides("openreg", OpenRegDeviceOpOverrides())
