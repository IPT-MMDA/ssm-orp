import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer


class SparseSSMPruner:
    """
    SparseSSM: One-shot training-free pruning for Mamba (Algorithm 1).

    Based on: "SparseSSM: Efficient Selective Structured State Space
    Models Can Be Pruned in One-Shot" (Tuo & Wang, arXiv:2506.09613)

    Key ideas:
      1) OBS importance for A_log:  I_{d,n} ∝ A²_log * Σ h²   (Theorem 1)
      2) Time-selective mask aggregation across sequence steps  (Algorithm 1)
      3) Sensitivity-aware FFN pruning                          (Section 3.4)
    """

    def __init__(self, model, calibration_dataset, args):
        self.model = model
        self.dataset = calibration_dataset
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_mixers(self):
        return [layer.mixer for layer in self.model.backbone.layers]

    def _analyze(self):
        print("\nModel Parameter Analysis:")
        groups = {}
        for name, p in self.model.named_parameters():
            for key in ["A_log", "in_proj", "out_proj", "conv1d",
                        "x_proj", "dt_proj", "embed", "norm"]:
                if key in name:
                    groups.setdefault(key, [0, 0])
                    groups[key][0] += 1
                    groups[key][1] += p.numel()
                    break
            else:
                groups.setdefault("other", [0, 0])
                groups["other"][0] += 1
                groups["other"][1] += p.numel()
        for cat, (cnt, total) in groups.items():
            print(f"  {cat}: {cnt} tensors, {total:,} params")

    # ------------------------------------------------------------------
    # Core: SSM selective-scan with statistics collection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _ssm_scan_collect(self, mixer, input_states, sparsity):
        """
        Replays the SSM selective scan on *input_states* (the mixer's
        input after layer-norm) and returns:
          h_squared_sum  [D, N]  – Σ_{b,t} h²  (for L2 baseline)
          C_count        [D, N]  – per-time-step candidate frequency
                                    (Algorithm 1, Phase 2)

        Vectorized: pre-computes all projections then runs the
        sequential scan with minimal per-step overhead.
        """
        batch_size, seq_len, _ = input_states.shape
        D = mixer.intermediate_size
        N = mixer.ssm_state_size
        total = D * N
        dev = input_states.device
        K = int(sparsity * total)

        # 1. in_proj → split x, z  (one matmul for full sequence)
        projected = mixer.in_proj(input_states).transpose(1, 2)   # [B,2D,L]
        hidden_states, _gate = projected.chunk(2, dim=1)          # [B,D,L]

        # 2. causal conv1d
        hidden_states = mixer.act(
            mixer.conv1d(hidden_states)[..., :seq_len]
        )                                                          # [B,D,L]

        # 3. x_proj → dt, B, C  (one matmul)
        ssm_params = mixer.x_proj(hidden_states.transpose(1, 2))  # [B,L,r+2N]
        dt_rank = mixer.time_step_rank
        time_step, B_ssm, _C_ssm = torch.split(
            ssm_params, [dt_rank, N, N], dim=-1
        )

        # 4. dt_proj + softplus  (vectorized over full sequence)
        dt = F.softplus(
            F.linear(time_step, mixer.dt_proj.weight, mixer.dt_proj.bias)
        ).transpose(1, 2)                                         # [B,D,L]

        # 5. Pre-compute dA and dB for ALL time steps at once
        # Capture A_log values NOW (before any potential in-place modification)
        # and clamp to reasonable range so pruned entries (A_log=+38) don't
        # dominate the importance scores in subsequent calibration chunks.
        A_log_float = mixer.A_log.float()
        # Mask out already-pruned entries (A_log >= 37) from importance scoring
        already_pruned = A_log_float >= 37.0
        A = -torch.exp(A_log_float)                                # [D,N]
        # For importance scoring, treat already-pruned entries as zero weight
        A_log_sq = A_log_float.clone()
        A_log_sq[already_pruned] = 0.0
        A_log_sq = A_log_sq ** 2                                   # [D,N]

        # dt: [B,D,L] → [B,D,L,1];  A: [D,N] → [1,D,1,N]
        dA_all = torch.exp(
            dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2)
        )                                                          # [B,D,L,N]
        # B_ssm: [B,L,N] → [B,1,L,N];  dt: [B,D,L] → [B,D,L,1]
        dB_all = (dt.unsqueeze(-1) *
                  B_ssm.unsqueeze(1).float())                      # [B,D,L,N]
        x_all = hidden_states.float().unsqueeze(-1)                # [B,D,L,1]

        # 6. Sequential scan (unavoidable recurrence) — tight loop
        #    No per-step topk here; candidate selection is batched after
        #    the loop to avoid ~5 Python→C++ dispatches per step.
        ssm_state = torch.zeros(batch_size, D, N, device=dev,
                                dtype=torch.float32)
        h_squared_sum = torch.zeros(D, N, device=dev,
                                    dtype=torch.float32)
        need_ccount = 0 < K < total
        if need_ccount:
            h_sq_all = torch.empty(seq_len, D, N, device=dev,
                                   dtype=torch.float32)

        for t in range(seq_len):
            ssm_state = (dA_all[:, :, t] * ssm_state +
                         dB_all[:, :, t] * x_all[:, :, t])         # [B,D,N]

            h_sq_t = ssm_state.square().sum(dim=0)                 # [D,N]
            h_squared_sum.add_(h_sq_t)
            if need_ccount:
                h_sq_all[t] = h_sq_t

        del dA_all, dB_all, x_all  # free ~200 MB before Phase 2

        # 7. Phase 2: batched candidate selection — single topk call
        #    over ALL time steps, then bincount for frequency.
        C_count = torch.zeros(D, N, device=dev, dtype=torch.float32)
        if need_ccount:
            h_sq_all *= A_log_sq.unsqueeze(0)               # in-place → M_all
            _, topk_idx = h_sq_all.view(seq_len, -1).topk(
                K, dim=1, largest=False, sorted=False
            )                                                # [L, K]
            C_count = torch.bincount(
                topk_idx.reshape(-1), minlength=total
            ).float().view(D, N)
            del h_sq_all, topk_idx

        return h_squared_sum, C_count

    # ------------------------------------------------------------------
    # SSM pruning  (Algorithm 1)
    # ------------------------------------------------------------------
    def prune_ssm(self):
        """
        Phases 1–3 of Algorithm 1:
          1. Collect hidden-state statistics from calibration data
          2. Per-time-step candidate selection (OBS importance)
          3. Final mask construction via frequency counting

        Optimization: runs the model forward pass once per chunk,
        capturing mixer inputs via hooks, then performs the SSM scan
        for statistics collection on captured tensors.
        """
        mixers = self._get_mixers()
        sparsity = self.args.sparsity
        n_layers = len(mixers)
        max_len = getattr(self.args, "max_length", 512)

        # accumulators per layer
        layer_h2 = [torch.zeros_like(m.A_log, dtype=torch.float32)
                     for m in mixers]
        layer_C  = [torch.zeros_like(m.A_log, dtype=torch.float32)
                     for m in mixers]

        # hooks to capture each mixer's input
        captured = {}
        hooks = []
        for i in range(n_layers):
            def _hook(idx):
                def fn(module, args):
                    captured[idx] = args[0].detach()
                return fn
            hooks.append(
                self.model.backbone.layers[i].mixer
                    .register_forward_pre_hook(_hook(i))
            )

        # Concatenate + tokenize + chunk calibration data
        # (matches paper: "128 contiguous segments of 2048 tokens")
        nsamples = getattr(self.args, "nsamples", 128)
        all_text = "\n\n".join(
            ex["text"].strip() for ex in self.dataset
            if ex["text"].strip()
        )
        encodings = self.tokenizer(all_text, return_tensors="pt")
        all_ids = encodings["input_ids"][0]  # [total_tokens]
        n_chunks = min(nsamples, all_ids.size(0) // max_len)
        all_ids = all_ids[: n_chunks * max_len]
        print(f"   Calibration: {all_ids.numel():,} tokens "
              f"-> {n_chunks} chunks of {max_len}")

        self.model.eval()
        import time as _time
        t0 = _time.perf_counter()
        for chunk_idx in tqdm(range(n_chunks), desc="Phase 1: Calibration"):
            input_ids = all_ids[
                chunk_idx * max_len : (chunk_idx + 1) * max_len
            ].unsqueeze(0).to(self.model.device)

            with torch.no_grad():
                self.model.backbone(input_ids)  # skip LM head

            for idx in range(n_layers):
                if idx in captured:
                    h2, C = self._ssm_scan_collect(
                        mixers[idx], captured[idx], sparsity
                    )
                    layer_h2[idx] += h2
                    layer_C[idx]  += C
            captured.clear()

        for h in hooks:
            h.remove()
        elapsed = _time.perf_counter() - t0
        print(f"   Collected from {n_chunks} chunks in {elapsed:.1f}s "
              f"({elapsed/n_chunks:.2f}s/chunk)")

        # Phase 3: construct masks and prune A_log
        total_pruned = 0
        total_params = 0
        use_alg1 = getattr(self.args, "ssm_method", "algorithm1") == "algorithm1"

        for layer_idx, mixer in enumerate(mixers):
            D, N = mixer.A_log.shape
            K = int(sparsity * D * N)
            total_params += D * N
            if K <= 0 or K >= D * N:
                continue

            if use_alg1:
                # Algorithm 1: prune entries with highest C (most
                # frequently selected as unimportant across time-steps)
                scores = layer_C[layer_idx]
                _, top_idx = torch.topk(scores.flatten(), K)
            else:
                # L2 baseline: prune entries with smallest OBS importance
                importance = (mixer.A_log.float() ** 2) * layer_h2[layer_idx]
                _, top_idx = torch.topk(importance.flatten(), K, largest=False)

            mask = torch.ones(D * N, device=mixer.A_log.device, dtype=torch.bool)
            mask[top_idx] = False
            mask = mask.reshape(D, N)

            # Diagnostic: show A_log stats before pruning (first/last layer)
            if layer_idx in (0, n_layers - 1):
                a = mixer.A_log.float()
                pruned_vals = a[~mask]
                kept_vals = a[mask]
                print(f"   [L{layer_idx}] A_log range: "
                      f"[{a.min():.3f}, {a.max():.3f}], "
                      f"pruned mean={pruned_vals.mean():.3f}, "
                      f"kept mean={kept_vals.mean():.3f}")
                if use_alg1:
                    sc = scores.flatten()
                    print(f"   [L{layer_idx}] C_count: min={sc.min():.0f}, "
                          f"max={sc.max():.0f}, "
                          f"mean={sc.mean():.1f}, "
                          f"zeros={int((sc == 0).sum())}")

            with torch.no_grad():
                # Setting A_log=0 gives A=-exp(0)=-1, which is WRONG:
                # the pruned SSM state becomes a near-integrator (dA≈0.97).
                # To truly silence a state we need dA=exp(dt*A)→0,
                # i.e. A→-∞, i.e. A_log→+∞.
                # We use a large finite value (+38) so fp16 exp doesn't NaN.
                LARGE = 38.0
                mixer.A_log.data[~mask] = LARGE
            total_pruned += K

        pct = total_pruned / total_params * 100 if total_params else 0
        print(f"   SSM: Pruned {total_pruned:,}/{total_params:,} A_log "
              f"entries ({pct:.1f}% sparsity)")
        return total_pruned

    # ------------------------------------------------------------------
    # FFN pruning  (Section 3.4, sensitivity-aware)
    # ------------------------------------------------------------------
    def prune_ffn(self):
        """
        Magnitude pruning of FFN components with sensitivity-aware
        sparsity allocation (Section 3.4, Eq. 7).

        Note: the paper uses SparseGPT (Hessian-aware weight reconstruction)
        for FFN pruning. Since we use simpler magnitude pruning, we apply
        a larger sensitivity gap to protect in_proj / out_proj which are
        the most sensitive modules (Table 8, Appendix B.2.2).
        """
        sparsity = self.args.sparsity
        total_pruned = 0
        total_params = 0

        # Per-module sparsity: in_proj/out_proj are extremely sensitive
        # to magnitude pruning (Table 2: MP gives PPL=7.2e13 on Mamba-130M).
        # Without SparseGPT's Hessian reconstruction, we must skip them
        # entirely and only prune robust modules.
        module_sparsity = {
            "conv1d":   min(1.0, sparsity),           # robust (Table 8)
            "x_proj":   min(1.0, sparsity),           # robust (Table 8)
            "dt_proj":  min(1.0, sparsity),           # robust (Table 8)
            "in_proj":  0.0,                          # skip: too sensitive
            "out_proj": 0.0,                          # skip: too sensitive
        }

        for layer in tqdm(self.model.backbone.layers, desc="Pruning FFN"):
            mixer = layer.mixer
            for mod_name, module in [("conv1d",   mixer.conv1d),
                                     ("x_proj",   mixer.x_proj),
                                     ("dt_proj",  mixer.dt_proj),
                                     ("in_proj",  mixer.in_proj),
                                     ("out_proj", mixer.out_proj)]:
                for pname, param in module.named_parameters():
                    if "weight" not in pname:
                        continue
                    s = module_sparsity[mod_name]

                    with torch.no_grad():
                        imp = torch.abs(param.data)
                        k = int(s * param.numel())
                        if 0 < k < param.numel():
                            thr = torch.kthvalue(imp.flatten(), k)[0]
                            param.data *= (imp > thr).to(param.dtype)
                            total_pruned += k
                    total_params += param.numel()

        pct = total_pruned / total_params * 100 if total_params else 0
        print(f"   FFN: Pruned {total_pruned:,}/{total_params:,} "
              f"({pct:.1f}% sparsity)")
        # Show per-module sparsity
        for mod, s in module_sparsity.items():
            print(f"     {mod}: {s*100:.0f}%")
        return total_pruned

    # ------------------------------------------------------------------
    # Structured pruning  (Section 4.3)
    # ------------------------------------------------------------------
    def prune_structured(self):
        """
        Structured pruning of SSM: remove entire columns of A_log
        (i.e. entire hidden-state channels of the SSM).

        From Section 4.3 of the paper:
          "We target the second axis of A: we aggregate the importance
           of each column by computing its L1 norm and then remove the
           least important columns. Simultaneously, we resize the output
           dimension of the linear x_proj layer to preserve tensor
           compatibility."

        This yields real inference speed-up (paper: 1.72x at 50%).
        """
        mixers = self._get_mixers()
        sparsity = self.args.sparsity
        n_layers = len(mixers)
        max_len = getattr(self.args, "max_length", 512)

        # Phase 1: collect hidden-state statistics (same as unstructured)
        layer_h2 = [torch.zeros_like(m.A_log, dtype=torch.float32)
                     for m in mixers]

        captured = {}
        hooks = []
        for i in range(n_layers):
            def _hook(idx):
                def fn(module, args):
                    captured[idx] = args[0].detach()
                return fn
            hooks.append(
                self.model.backbone.layers[i].mixer
                    .register_forward_pre_hook(_hook(i))
            )

        # Same chunk-based tokenization as prune_ssm — O(nsamples) not O(dataset)
        nsamples = getattr(self.args, "nsamples", 128)
        all_text = "\n\n".join(
            ex["text"].strip() for ex in self.dataset
            if ex["text"].strip()
        )
        encodings = self.tokenizer(all_text, return_tensors="pt")
        all_ids = encodings["input_ids"][0]
        n_chunks = min(nsamples, all_ids.size(0) // max_len)
        all_ids = all_ids[: n_chunks * max_len]
        print(f"   Calibration: {all_ids.numel():,} tokens "
              f"-> {n_chunks} chunks of {max_len}")

        self.model.eval()
        for chunk_idx in tqdm(range(n_chunks), desc="Structured: Calibration"):
            input_ids = all_ids[
                chunk_idx * max_len : (chunk_idx + 1) * max_len
            ].unsqueeze(0).to(self.model.device)

            with torch.no_grad():
                self.model.backbone(input_ids)

            for idx in range(n_layers):
                if idx in captured:
                    h2, _ = self._ssm_scan_collect(
                        mixers[idx], captured[idx], sparsity
                    )
                    layer_h2[idx] += h2
            captured.clear()

        for h in hooks:
            h.remove()
        print(f"   Collected statistics from {n_chunks} chunks")

        # Phase 2: for each layer, compute column importance and prune
        total_cols_pruned = 0
        total_cols = 0

        for layer_idx, mixer in enumerate(mixers):
            D, N = mixer.A_log.shape
            K = int(sparsity * N)  # number of columns to remove
            total_cols += N
            if K <= 0 or K >= N:
                continue

            # Column importance: A²_log * h² aggregated per column (L1 norm)
            importance = (mixer.A_log.float() ** 2) * layer_h2[layer_idx]
            col_importance = importance.sum(dim=0)  # [N]

            # Find columns to keep
            _, keep_idx = torch.topk(col_importance, N - K, largest=True)
            keep_idx = keep_idx.sort()[0]  # keep sorted order

            with torch.no_grad():
                # 1. Prune A_log: [D, N] -> [D, N-K]
                mixer.A_log = torch.nn.Parameter(
                    mixer.A_log.data[:, keep_idx]
                )

                # 2. Prune D parameter if it exists and has N dimension
                if hasattr(mixer, 'D') and mixer.D is not None:
                    if mixer.D.numel() == D:
                        pass  # D param is per-channel D, not per-state N
                    elif mixer.D.shape[-1] == N:
                        mixer.D = torch.nn.Parameter(
                            mixer.D.data[..., keep_idx]
                        )

                # 3. Resize x_proj: output produces [dt_rank, B, C]
                #    B and C have dimension N each, so we need to adjust
                #    the last 2*N columns of x_proj.weight
                dt_rank = mixer.time_step_rank
                w = mixer.x_proj.weight.data  # [dt_rank + 2N, D_in]
                # Split into dt, B, C parts
                w_dt = w[:dt_rank, :]           # [dt_rank, D_in]
                w_B  = w[dt_rank:dt_rank+N, :]  # [N, D_in]
                w_C  = w[dt_rank+N:, :]         # [N, D_in]
                # Keep only selected columns
                w_B_new = w_B[keep_idx, :]      # [N-K, D_in]
                w_C_new = w_C[keep_idx, :]      # [N-K, D_in]
                new_w = torch.cat([w_dt, w_B_new, w_C_new], dim=0)
                new_out = new_w.shape[0]
                new_proj = torch.nn.Linear(
                    w.shape[1], new_out, bias=False,
                    device=w.device, dtype=w.dtype
                )
                new_proj.weight = torch.nn.Parameter(new_w)
                mixer.x_proj = new_proj

                # 4. Update ssm_state_size so the slow-path scan uses
                #    the correct N dimension
                mixer.ssm_state_size = N - K

            total_cols_pruned += K

        pct = total_cols_pruned / total_cols * 100 if total_cols else 0
        print(f"   Structured: Removed {total_cols_pruned}/{total_cols} "
              f"SSM columns ({pct:.1f}% structured sparsity)")

        # Count actual parameter reduction
        new_total = sum(p.numel() for p in self.model.parameters())
        return total_cols_pruned

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prune(self):
        """Full SparseSSM pipeline."""
        method = getattr(self.args, "ssm_method", "algorithm1")
        mode   = getattr(self.args, "prune_mode", "ssm")
        print(f"SparseSSM Pruning | method={method} | mode={mode}")
        self._analyze()

        if mode in ("structured", "structured+ffn"):
            params_before = sum(p.numel() for p in self.model.parameters())
            print(f"\nStep 1: Structured Pruning of SSM "
                  f"({self.args.sparsity*100:.0f}% columns)...")
            self.prune_structured()

            if mode == "structured+ffn":
                print(f"\nStep 2: Pruning FFN (sensitivity-aware, "
                      f"{self.args.sparsity*100:.0f}%)...")
                self.prune_ffn()

            params_after = sum(p.numel() for p in self.model.parameters())
            nonzero = sum(p.count_nonzero().item() for p in self.model.parameters())
            removed = params_before - params_after
            zeroed = params_after - nonzero
            print(f"\nParams: {params_before:,} → {params_after:,} "
                  f"(-{removed:,} removed, {zeroed:,} zeroed)")
            print(f"   Effective reduction: "
                  f"{(removed + zeroed) / params_before * 100:.1f}%")
            return self.model

        print(f"\nStep 1: Pruning SSM A_log ({self.args.sparsity*100:.0f}% "
              f"sparsity, {method})...")
        ssm_pruned = self.prune_ssm()

        ffn_pruned = 0
        if mode == "full":
            print(f"\nStep 2: Pruning FFN (sensitivity-aware)...")
            ffn_pruned = self.prune_ffn()

        total_all = sum(p.numel() for p in self.model.parameters())
        total_pruned = ssm_pruned + ffn_pruned
        print(f"\nTotal: {total_pruned:,}/{total_all:,} pruned "
              f"({total_pruned/total_all*100:.2f}% overall)")
        return self.model