# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .vmfb_comparison import run_vmfb_comparison

def test_end_to_end_cpu():
    run_vmfb_comparison(device="llvm-cpu", precision="fp32", quantization="unquantized")