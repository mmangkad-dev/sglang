MiB = 1024 * 1024

TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES = {
    # H100 SXM NVSwitch sweeps found no stable eager or graph win over the
    # fastest custom-AR, PyNCCL, or NCCL symmetric-memory fallback.
    9: {},
    10: {
        2: 64 * MiB,  # 64 MB
        4: 64 * MiB,  # 64 MB
        6: 128 * MiB,  # 128 MB
        8: 128 * MiB,  # 128 MB
    },
}
