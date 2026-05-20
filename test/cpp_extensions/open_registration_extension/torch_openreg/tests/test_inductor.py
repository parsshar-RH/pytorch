# Owner(s): ["module: PrivateUse1"]

import torch
import torch._dynamo
from torch.testing._internal.common_utils import run_tests, TestCase


class TestOpenRegInductorBackend(TestCase):
    def test_scheduling_registered(self):
        from torch._inductor.codegen.common import get_scheduling_for_device

        sched = get_scheduling_for_device("openreg")
        self.assertIsNotNone(sched)

    def test_device_op_overrides_registered(self):
        from torch._inductor.codegen.common import get_device_op_overrides

        overrides = get_device_op_overrides("openreg")
        self.assertIsNotNone(overrides)

    def test_inductor_wrapper_codegen_class(self):
        from torch._inductor.codegen.common import get_wrapper_codegen_for_device

        from torch_openreg.inductor import OpenRegWrapperCodegen

        wrapper_cls = get_wrapper_codegen_for_device("openreg")
        self.assertIs(wrapper_cls, OpenRegWrapperCodegen)

    def test_device_op_overrides_methods(self):
        from torch._inductor.codegen.common import get_device_op_overrides

        overrides = get_device_op_overrides("openreg")
        self.assertIn("get_raw_stream", overrides.import_get_raw_stream_as("get_raw_stream"))
        self.assertIn("_set_device", overrides.set_device(0))
        self.assertEqual(overrides.synchronize(), "pass")

    def test_inductor_scheduling_class(self):
        from torch._inductor.codegen.common import get_scheduling_for_device
        from torch._inductor.codegen.cpp import CppScheduling

        sched = get_scheduling_for_device("openreg")
        self.assertIs(sched, CppScheduling)

    def test_inductor_simple_compile(self):
        @torch.compile(backend="inductor")
        def fn(x):
            return x + 1

        x = torch.randn(4, device="openreg")
        result = fn(x)
        self.assertEqual(result, x + 1)

    def test_inductor_matches_eager(self):
        def fn(x):
            return x * 2 + x

        x = torch.randn(4, device="openreg")
        compiled = torch.compile(fn, backend="inductor")
        self.assertEqual(compiled(x), fn(x))


if __name__ == "__main__":
    run_tests()
