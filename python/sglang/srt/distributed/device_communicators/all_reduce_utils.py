MiB = 1024 * 1024

TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES = {
    # H100 and H200 NVSwitch sweeps found no stable eager or graph win over
    # the fastest custom-AR, PyNCCL, or NCCL symmetric-memory fallback.
    9: {},
    # B200 and B300 NVSwitch sweeps likewise found no stable win over the
    # fastest fallback at TP2, TP4, TP6, or TP8.
    10: {},
}
