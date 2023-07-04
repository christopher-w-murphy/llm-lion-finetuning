import ctypes as ct

from bitsandbytes.cextension import COMPILED_WITH_CUDA, lib
from bitsandbytes.functional import prod, get_ptr, pre_call, post_call, is_on_gpu
from numpy import ctypeslib
import torch
from torch import Tensor

if COMPILED_WITH_CUDA:
    """C FUNCTIONS FOR OPTIMIZERS"""
    str2optimizer32bit = {}
    str2optimizer32bit["adam"] = (lib.cadam32bit_grad_fp32, lib.cadam32bit_grad_fp16, lib.cadam32bit_grad_bf16)
    str2optimizer32bit["momentum"] = (
        lib.cmomentum32bit_grad_32,
        lib.cmomentum32bit_grad_16,
    )
    str2optimizer32bit["rmsprop"] = (
        lib.crmsprop32bit_grad_32,
        lib.crmsprop32bit_grad_16,
    )
    str2optimizer32bit["lion"] = (lib.clion32bit_grad_fp32, lib.clion32bit_grad_fp16, lib.clion32bit_grad_bf16)
    str2optimizer32bit["adagrad"] = (
        lib.cadagrad32bit_grad_32,
        lib.cadagrad32bit_grad_16,
    )

    str2optimizer8bit = {}
    str2optimizer8bit["adam"] = (
        lib.cadam_static_8bit_grad_32,
        lib.cadam_static_8bit_grad_16,
    )
    str2optimizer8bit["momentum"] = (
        lib.cmomentum_static_8bit_grad_32,
        lib.cmomentum_static_8bit_grad_16,
    )
    str2optimizer8bit["rmsprop"] = (
        lib.crmsprop_static_8bit_grad_32,
        lib.crmsprop_static_8bit_grad_16,
    )
    str2optimizer8bit["lion"] = (
        lib.clion_static_8bit_grad_32,
        lib.clion_static_8bit_grad_16,
    )
    str2optimizer8bit["lamb"] = (
        lib.cadam_static_8bit_grad_32,
        lib.cadam_static_8bit_grad_16,
    )
    str2optimizer8bit["lars"] = (
        lib.cmomentum_static_8bit_grad_32,
        lib.cmomentum_static_8bit_grad_16,
    )

    str2optimizer8bit_blockwise = {}
    str2optimizer8bit_blockwise["adam"] = (
        lib.cadam_8bit_blockwise_grad_fp32,
        lib.cadam_8bit_blockwise_grad_fp16,
        lib.cadam_8bit_blockwise_grad_bf16,
    )
    str2optimizer8bit_blockwise["momentum"] = (
        lib.cmomentum_8bit_blockwise_grad_fp32,
        lib.cmomentum_8bit_blockwise_grad_fp16,
    )
    str2optimizer8bit_blockwise["rmsprop"] = (
        lib.crmsprop_8bit_blockwise_grad_fp32,
        lib.crmsprop_8bit_blockwise_grad_fp16,
    )
    str2optimizer8bit_blockwise["lion"] = (
        lib.clion_8bit_blockwise_grad_fp32,
        lib.clion_8bit_blockwise_grad_fp16,
        lib.clion_8bit_blockwise_grad_bf16,
    )
    str2optimizer8bit_blockwise["adagrad"] = (
        lib.cadagrad_8bit_blockwise_grad_fp32,
        lib.cadagrad_8bit_blockwise_grad_fp16,
    )

dtype2bytes = {}
dtype2bytes[torch.float32] = 4
dtype2bytes[torch.float16] = 2
dtype2bytes[torch.bfloat16] = 2
dtype2bytes[torch.uint8] = 1
dtype2bytes[torch.int8] = 1


def elementwise_func(func_name, A, B, value, prefetch=True):
    func = None
    if A.dtype == torch.float32:
        func = getattr(lib, f'c{func_name}_fp32', None)
        cvalue = ct.c_float(value)
    elif A.dtype == torch.uint8:
        func = getattr(lib, f'c{func_name}_uint8', None)
        cvalue = ct.c_uint8(value)

    if func is None: raise NotImplementedError(f'Function not implemented: {func_name}')

    is_managed = getattr(A, 'is_managed', False)
    if is_managed and prefetch:
        prefetch_tensor(A)
        if B is not None: prefetch_tensor(B)

    func(get_ptr(A), get_ptr(B), cvalue, ct.c_int64(A.numel()))
    if A.is_paged or B.is_paged:
        # paged function are fully asynchronous
        # if we return from this function, we want to the tensor
        # to be in the correct state, that is the final state after the
        # operation occured. So we synchronize.
        torch.cuda.synchronize()


class GlobalPageManager:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.paged_tensors = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def prefetch_all(self, to_cpu=False):
        # assume the first added, will be hte
        # ones that are used first, so swap them in last
        # in the case they are evicted again
        for t in self.paged_tensors[::-1]:
            prefetch_tensor(t, to_cpu)


def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    """
    Creates the dynamic quantiztion map.

    The dynamic data type is made up of a dynamic exponent and
    fraction. As the exponent increase from 0 to -7 the number
    of bits available for the fraction shrinks.

    This is a generalization of the dynamic type where a certain
    number of the bits and be reserved for the linear quantization
    region (the fraction). n determines the maximum number of
    exponent bits.

    For more details see
    (8-Bit Approximations for Parallelism in Deep Learning)[https://arxiv.org/abs/1511.04561]
    """

    data = []
    # these are additional items that come from the case
    # where all the exponent bits are zero and no
    # indicator bit is present
    non_sign_bits = total_bits - (1 if signed else 0)
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    if not signed:
        additional_items = 2 * additional_items
    for i in range(max_exponent_bits):
        fraction_items = int((2 ** (i + non_sign_bits - max_exponent_bits) + 1 if signed else 2 ** (
                i + non_sign_bits - max_exponent_bits + 1) + 1))
        boundaries = torch.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

        if additional_items > 0:
            boundaries = torch.linspace(0.1, 1, additional_items + 1)
            means = (boundaries[:-1] + boundaries[1:]) / 2.0
            data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
            if signed:
                data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    data.append(0)
    data.append(1.0)

    gap = 256 - len(data)
    for i in range(gap):
        data.append(0)

    data.sort()
    return Tensor(data)


def get_paged(*shape, dtype=torch.float32, device=torch.device('cuda', index=0)):
    num_bytes = dtype2bytes[dtype] * prod(shape)
    cuda_ptr = lib.cget_managed_ptr(ct.c_size_t(num_bytes))
    c_ptr = ct.cast(cuda_ptr, ct.POINTER(ct.c_int))
    new_array = ctypeslib.as_array(c_ptr, shape=shape)
    out = torch.frombuffer(new_array, dtype=dtype, count=prod(shape)).view(shape)
    out.is_paged = True
    out.page_deviceid = device.index
    return out


def prefetch_tensor(A, to_cpu=False):
    assert A.is_paged, 'Only paged tensors can be prefetched!'
    if to_cpu:
        deviceid = -1
    else:
        deviceid = A.page_deviceid

    num_bytes = dtype2bytes[A.dtype] * A.numel()
    lib.cprefetch(get_ptr(A), ct.c_size_t(num_bytes), ct.c_int32(deviceid))


def fill(A, value, device=None, prefetch=True):
    elementwise_func('fill', A, None, value)


def percentile_clipping(
        grad: Tensor, gnorm_vec: Tensor, step: int, percentile: int = 5
):
    """Applies percentile clipping

    grad: torch.Tensor
        The gradient tensor.
    gnorm_vec: torch.Tensor
        Vector of gradient norms. 100 elements expected.
    step: int
        The current optimiation steps (number of past gradient norms).

    """
    prev_device = pre_call(grad.device)
    is_on_gpu([grad, gnorm_vec])
    if grad.dtype == torch.float32:
        lib.cpercentile_clipping_g32(
            get_ptr(grad),
            get_ptr(gnorm_vec),
            ct.c_int32(step),
            ct.c_int32(grad.numel()),
        )
    elif grad.dtype == torch.float16:
        lib.cpercentile_clipping_g16(
            get_ptr(grad),
            get_ptr(gnorm_vec),
            ct.c_int32(step),
            ct.c_int32(grad.numel()),
        )
    else:
        raise ValueError(f"Gradient type {grad.dtype} not supported!")
    post_call(prev_device)

    current_gnorm = torch.sqrt(gnorm_vec[step % 100])
    vals, idx = torch.sort(gnorm_vec)
    clip_value = torch.sqrt(vals[percentile])
    gnorm_scale = 1.0

    if current_gnorm > clip_value:
        gnorm_scale = clip_value / current_gnorm

    return current_gnorm, clip_value, gnorm_scale


def optimizer_update_32bit(
        optimizer_name: str,
        g: Tensor,
        p: Tensor,
        state1: Tensor,
        beta1: float,
        eps: float,
        step: int,
        lr: float,
        state2: Tensor = None,
        beta2: float = 0.0,
        weight_decay: float = 0.0,
        gnorm_scale: float = 1.0,
        unorm_vec: Tensor = None,
        max_unorm: float = 0.0,
        skip_zeros=False,
) -> None:
    """
    Performs an inplace optimizer update with one or two optimizer states.

    Universal optimizer update for 32-bit state and 32/16-bit gradients/weights.

    Parameters
    ----------
    optimizer_name : str
        The name of the optimizer: {adam}.
    g : torch.Tensor
        Gradient tensor.
    p : torch.Tensor
        Parameter tensor.
    state1 : torch.Tensor
        Optimizer state 1.
    beta1 : float
        Optimizer beta1.
    eps : float
        Optimizer epsilon.
    weight_decay : float
        Weight decay.
    step : int
        Current optimizer step.
    lr : float
        The learning rate.
    state2 : torch.Tensor
        Optimizer state 2.
    beta2 : float
        Optimizer beta2.
    gnorm_scale : float
        The factor to rescale the gradient to the max clip value.
    unorm_vec : torch.Tensor
        The tensor for the update norm.
    max_unorm : float
        The maximum update norm relative to the weight norm.
    skip_zeros : bool
        Whether to skip zero-valued gradients or not (default: False).
    """

    param_norm = 0.0
    if max_unorm > 0.0:
        param_norm = torch.norm(p.data.float())

    optim_func = None
    if g.dtype == torch.float32:
        optim_func = str2optimizer32bit[optimizer_name][0]
    elif g.dtype == torch.float16:
        optim_func = str2optimizer32bit[optimizer_name][1]
    elif (g.dtype == torch.bfloat16 and len(str2optimizer32bit[optimizer_name]) == 3):
        optim_func = str2optimizer32bit[optimizer_name][2]
    else:
        raise ValueError(
            f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}")

    is_on_gpu([g, p, state1, state2, unorm_vec])
    prev_device = pre_call(g.device)
    optim_func(
        get_ptr(g),
        get_ptr(p),
        get_ptr(state1),
        get_ptr(state2),
        get_ptr(unorm_vec),
        ct.c_float(max_unorm),
        ct.c_float(param_norm),
        ct.c_float(beta1),
        ct.c_float(beta2),
        ct.c_float(eps),
        ct.c_float(weight_decay),
        ct.c_int32(step),
        ct.c_float(lr),
        ct.c_float(gnorm_scale),
        ct.c_bool(skip_zeros),
        ct.c_int32(g.numel()))
    post_call(prev_device)


def optimizer_update_8bit(
        optimizer_name: str,
        g: Tensor,
        p: Tensor,
        state1: Tensor,
        state2: Tensor,
        beta1: float,
        beta2: float,
        eps: float,
        step: int,
        lr: float,
        qmap1: Tensor,
        qmap2: Tensor,
        max1: Tensor,
        max2: Tensor,
        new_max1: Tensor,
        new_max2: Tensor,
        weight_decay: float = 0.0,
        gnorm_scale: float = 1.0,
        unorm_vec: Tensor = None,
        max_unorm: float = 0.0,
) -> None:
    """
    Performs an inplace Adam update.

    Universal Adam update for 32/8-bit state and 32/16-bit gradients/weights.
    Uses AdamW formulation if weight decay > 0.0.

    Parameters
    ----------
    optimizer_name : str
        The name of the optimizer. Choices {adam, momentum}
    g : torch.Tensor
        Gradient tensor.
    p : torch.Tensor
        Parameter tensor.
    state1 : torch.Tensor
        Adam state 1.
    state2 : torch.Tensor
        Adam state 2.
    beta1 : float
        Adam beta1.
    beta2 : float
        Adam beta2.
    eps : float
        Adam epsilon.
    weight_decay : float
        Weight decay.
    step : int
        Current optimizer step.
    lr : float
        The learning rate.
    qmap1 : torch.Tensor
        Quantization map for first Adam state.
    qmap2 : torch.Tensor
        Quantization map for second Adam state.
    max1 : torch.Tensor
        Max value for first Adam state update.
    max2 : torch.Tensor
        Max value for second Adam state update.
    new_max1 : torch.Tensor
        Max value for the next Adam update of the first state.
    new_max2 : torch.Tensor
        Max value for the next Adam update of the second state.
    gnorm_scale : float
        The factor to rescale the gradient to the max clip value.
    unorm_vec : torch.Tensor
        The tensor for the update norm.
    max_unorm : float
        The maximum update norm relative to the weight norm.
    """

    param_norm = 0.0
    if max_unorm > 0.0:
        param_norm = torch.norm(p.data.float())

    prev_device = pre_call(g.device)
    is_on_gpu([g, p, state1, state2, unorm_vec, qmap1, qmap2, max1, max2, new_max1, new_max2])
    if g.dtype == torch.float32 and state1.dtype == torch.uint8:
        str2optimizer8bit[optimizer_name][0](
            get_ptr(p),
            get_ptr(g),
            get_ptr(state1),
            get_ptr(state2),
            get_ptr(unorm_vec),
            ct.c_float(max_unorm),
            ct.c_float(param_norm),
            ct.c_float(beta1),
            ct.c_float(beta2),
            ct.c_float(eps),
            ct.c_int32(step),
            ct.c_float(lr),
            get_ptr(qmap1),
            get_ptr(qmap2),
            get_ptr(max1),
            get_ptr(max2),
            get_ptr(new_max1),
            get_ptr(new_max2),
            ct.c_float(weight_decay),
            ct.c_float(gnorm_scale),
            ct.c_int32(g.numel()),
        )
    elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
        str2optimizer8bit[optimizer_name][1](
            get_ptr(p),
            get_ptr(g),
            get_ptr(state1),
            get_ptr(state2),
            get_ptr(unorm_vec),
            ct.c_float(max_unorm),
            ct.c_float(param_norm),
            ct.c_float(beta1),
            ct.c_float(beta2),
            ct.c_float(eps),
            ct.c_int32(step),
            ct.c_float(lr),
            get_ptr(qmap1),
            get_ptr(qmap2),
            get_ptr(max1),
            get_ptr(max2),
            get_ptr(new_max1),
            get_ptr(new_max2),
            ct.c_float(weight_decay),
            ct.c_float(gnorm_scale),
            ct.c_int32(g.numel()),
        )
    else:
        raise ValueError(
            f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}"
        )
    post_call(prev_device)


def optimizer_update_8bit_blockwise(
        optimizer_name: str,
        g: Tensor,
        p: Tensor,
        state1: Tensor,
        state2: Tensor,
        beta1: float,
        beta2: float,
        eps: float,
        step: int,
        lr: float,
        qmap1: Tensor,
        qmap2: Tensor,
        absmax1: Tensor,
        absmax2: Tensor,
        weight_decay: float = 0.0,
        gnorm_scale: float = 1.0,
        skip_zeros=False,
) -> None:
    optim_func = None
    prev_device = pre_call(g.device)
    is_on_gpu([g, p, state1, state2, qmap1, qmap2, absmax1, absmax2])
    if g.dtype == torch.float32 and state1.dtype == torch.uint8:
        optim_func = str2optimizer8bit_blockwise[optimizer_name][0]
    elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
        optim_func = str2optimizer8bit_blockwise[optimizer_name][1]
    elif (g.dtype == torch.bfloat16 and state1.dtype == torch.uint8 and
          len(str2optimizer8bit_blockwise[optimizer_name]) == 3):
        optim_func = str2optimizer8bit_blockwise[optimizer_name][2]
    else:
        raise ValueError(
            f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}"
        )
    post_call(prev_device)

    is_on_gpu([p, g, state1, state2, qmap1, qmap2, absmax1, absmax2])

    prev_device = pre_call(g.device)
    optim_func(
        get_ptr(p),
        get_ptr(g),
        get_ptr(state1),
        get_ptr(state2),
        ct.c_float(beta1),
        ct.c_float(beta2),
        ct.c_float(eps),
        ct.c_int32(step),
        ct.c_float(lr),
        get_ptr(qmap1),
        get_ptr(qmap2),
        get_ptr(absmax1),
        get_ptr(absmax2),
        ct.c_float(weight_decay),
        ct.c_float(gnorm_scale),
        ct.c_bool(skip_zeros),
        ct.c_int32(g.numel()),
    )
    post_call(prev_device)
