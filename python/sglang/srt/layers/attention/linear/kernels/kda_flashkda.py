from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

# FlashKDA chunk size. Sequences shorter than this fall back to Triton.
_FLASHKDA_CHUNK_SIZE = 64

# Length below which FlashKDA wins at any head count, so no occupancy check is
# needed.
_FLASHKDA_SHORT_SEQ_LEN = 2048

# Past _FLASHKDA_SHORT_SEQ_LEN, FlashKDA needs at least this many CTAs
# (longest-sequence equivalents x per-rank heads) to beat Triton.
_FLASHKDA_MIN_LONG_SEQ_CTAS = 32


def _load_flash_kda():
    """Import the optional ``flash_kda`` CUTLASS module."""
    try:
        import flash_kda
    except ImportError as e:
        raise ImportError(
            "The 'flashkda' KDA prefill backend requires the flash_kda module, "
            "which is not installed. Install it from source:\n"
            "    pip install git+https://github.com/MoonshotAI/FlashKDA.git"
        ) from e
    return flash_kda


def _triton_fallback(
    q,
    k,
    v,
    g,
    beta,
    ssm_states,
    cache_indices,
    query_start_loc,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    beta_is_raw=False,
    return_intermediate_states=False,
    track_state=None,
    track_chunk_idx=None,
):
    """Fall back to the Triton chunk_kda kernel (handles all preprocessing).

    `g` is the RAW gate; chunk_kda applies the gate activation internally when
    A_log is provided, so A_log/dt_bias/lower_bound must be threaded through too
    -- otherwise the fallback silently skips activation. chunk_kda updates the
    ssm state in-place via cache_indices and returns only the output tensor
    (or (output, h) when return_intermediate_states is set).
    """
    from sglang.kernels.ops.attention.fla.kda import chunk_kda

    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=ssm_states,
        initial_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=query_start_loc,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        beta_is_raw=beta_is_raw,
        output_intermediate_states=return_intermediate_states,
        track_state=track_state,
        track_chunk_idx=track_chunk_idx,
    )


class FlashKDAKernel(LinearAttnKernelBase):
    """FlashKDA (MoonshotAI) fully-fused CUTLASS KDA prefill backend.

    Wraps the external ``flash_kda`` package (https://github.com/MoonshotAI/FlashKDA).

    FlashKDA fuses q/k L2 norm, beta sigmoid, and the KDA gate *inside* the
    kernel, so we pass RAW tensors plus ``A_log``/``dt_bias``/``lower_bound``.
    It is prefill-only, bf16, K == V == 128, HV == H (no GVA), and requires the
    safe (bounded) gate (``lower_bound`` set). The non-safe path and sequences
    outside [chunk_size, max_seq_len] fall back to Triton ``chunk_kda``.
    Requires an SM90+ GPU with the ``flash_kda`` package.
    """

    # Tracked batches always take the Triton fallback, which forwards the
    # fp32 snapshot arguments (see _triton_fallback).
    supports_track_state_snapshot: bool = True

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("FlashKDAKernel only supports prefill (extend)")

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        extend_seq_lens_cpu: Optional[list] = None,
        is_spec_decode: bool = False,
        beta_is_raw: bool = False,
        return_intermediate_states: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # The fused kernel cannot expose per-chunk states (h), which the mamba
        # radix extra_buffer track path needs; route tracked batches through
        # the Triton chunk_kda fallback instead of silently skipping the
        # snapshot (that would corrupt prefix-cache restores).
        if return_intermediate_states or self._should_fall_back(
            lower_bound,
            is_spec_decode,
            query_start_loc,
            extend_seq_lens_cpu,
            # Grid is (sequences x heads); q is [1, packed_seq, H, K].
            q.shape[2],
        ):
            return _triton_fallback(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                beta_is_raw=beta_is_raw,
                return_intermediate_states=return_intermediate_states,
                track_state=kwargs.get("track_state"),
                track_chunk_idx=kwargs.get("track_chunk_idx"),
            )

        # Bare tensor, matching chunk_kda and the other KDA extend kernels: the
        # caller only unpacks (output, h) when it asked for intermediate states,
        # and that request always takes the Triton fallback above.
        return self._flashkda_extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            beta_is_raw=beta_is_raw,
        )

    @staticmethod
    def _should_fall_back(
        lower_bound: Optional[float],
        is_spec_decode: bool,
        query_start_loc: torch.Tensor,
        extend_seq_lens_cpu: Optional[list],
        num_heads: int,
    ) -> bool:
        """Whether to use the Triton chunk_kda path instead of the fused kernel."""
        # Safe-gate only: the fused kernel does not support the unbounded gate
        # (-exp(A_log)*softplus); those models leave lower_bound unset.
        if lower_bound is None:
            return True
        # FlashKDA writes the committed recurrent state back in place, so it is
        # unsafe for speculative verify / draft-extend forwards (which must stay
        # rollback-able). Those reach this backend through forward_extend, so
        # gate them here rather than relying on the decode/target_verify stubs.
        if is_spec_decode:
            return True
        # Read the per-request lengths from the CPU-side extend_seq_lens to
        # avoid a GPU->CPU sync on every layer; derive from query_start_loc
        # (one sync) only if they are unavailable.
        if extend_seq_lens_cpu is not None:
            if torch.is_tensor(extend_seq_lens_cpu):
                lo = int(extend_seq_lens_cpu.min())
                hi = int(extend_seq_lens_cpu.max())
                total = int(extend_seq_lens_cpu.sum())
            else:
                lo = min(extend_seq_lens_cpu)
                hi = max(extend_seq_lens_cpu)
                total = sum(extend_seq_lens_cpu)
        else:
            seq_lens = query_start_loc[1:] - query_start_loc[:-1]
            lo_t, hi_t = torch.aminmax(seq_lens)
            lo, hi, total = (
                int(x) for x in torch.stack((lo_t, hi_t, seq_lens.sum())).tolist()
            )
        # Sequences below the chunk size are faster on Triton.
        if lo < _FLASHKDA_CHUNK_SIZE:
            return True
        # Short sequences are a FlashKDA win at every head count measured.
        if hi <= _FLASHKDA_SHORT_SEQ_LEN:
            return False
        # Past that, FlashKDA parallelizes over (sequences x heads) and its cost
        # tracks the LONGEST sequence, while Triton chunk_kda also parallelizes
        # over chunks within a sequence so its cost tracks TOTAL tokens. The
        # fused kernel therefore wins only once the grid is populated enough;
        # "sequences" here is total/hi, so that a batch padded out with short
        # requests does not count as full. Measured on GB300 (D=128, bf16),
        # speedup vs Triton at 8192 tokens collapses onto CTAs = seqs x heads
        # regardless of how they split (H=4/8/16/32 all on one curve):
        #   4 CTAs 0.55x | 8 0.61x | 16 0.77-0.83x | 32 1.06-1.14x
        #   64 1.53-1.68x | 128 2.50-2.75x | 256 2.98x
        # so the crossover sits at _FLASHKDA_MIN_LONG_SEQ_CTAS.
        return total * num_heads < _FLASHKDA_MIN_LONG_SEQ_CTAS * hi

    def _flashkda_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        beta_is_raw: bool = False,
    ) -> torch.Tensor:
        flash_kda = _load_flash_kda()

        # Input shapes (varlen, B == 1, matching chunk_kda's contract):
        #   q, k = [1, packed_seq, H, K]   v = [1, packed_seq, HV, V]
        #   g    = [1, packed_seq, HV, K]  beta = [1, packed_seq, H]
        # flash_kda wants these 4D tensors directly and RAW (it fuses l2norm /
        # beta sigmoid / gate activation in-kernel).
        num_heads = q.shape[2]
        head_dim = q.shape[3]
        scale = head_dim**-0.5

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()

        # FlashKDA applies sigmoid internally; invert only the already-activated
        # Kimi beta path.
        if not beta_is_raw:
            beta = torch.logit(beta.float().clamp_(1e-7, 1.0 - 1e-7))
        beta = beta.to(torch.bfloat16).contiguous()

        # flash_kda wants A_log [H] fp32 and dt_bias [H, K] fp32. The model
        # stores A_log as [1, 1, H, 1] and dt_bias as 1D [H*K], so reshape both.
        A_log = A_log.reshape(-1).float().contiguous()
        if dt_bias is not None:
            dt_bias = dt_bias.reshape(num_heads, -1).float().contiguous()

        # cu_seqlens must be int64 for flash_kda (FLA casts to long).
        cu_seqlens = query_start_loc.to(torch.int64)

        # flash_kda varlen state is [N, H, V, K] -- the SAME layout as sglang's
        # KDA pool, so no transpose is needed. Advanced indexing copies, so the
        # final state is written back in-place below (matching chunk_kda).
        initial_state = ssm_states[cache_indices].contiguous()

        out_buf = torch.empty_like(v)
        final_state = torch.empty_like(initial_state)

        flash_kda.fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            out_buf,
            A_log,
            dt_bias,
            lower_bound,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
        )

        ssm_states[cache_indices] = final_state

        # out_buf is already [1, packed_seq, HV, V].
        return out_buf
