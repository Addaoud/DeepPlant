from typing import Optional, Any
import inspect
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
)
from peft import IA3Config, get_peft_model, TaskType

from src.seed import set_seed

set_seed()


MODEL_REGISTRY = {
    "nt": "InstaDeepAI/NTv3_650M_pre",
    "plantcad2": "kuleshov-group/PlantCAD2-Small-l24-d0768",
    "agront": "InstaDeepAI/agro-nucleotide-transformer-1b",
    "pdllm": "zhangtaolab/plant-dnagpt-6mer",
}


# Per-architecture IA3 target/feedforward module name hints. Confirm these with
# inspect_linear_modules.py before trusting them (print_trainable_parameters()
# must be non-zero after get_peft_model). The PlantCAD2 set matches the target
# modules the authors use in their own LoRA fine-tuning script.
IA3_TARGET_HINTS = {
    "nt": {
        "target_modules": [
            "query_head.linear",
            "key_head.linear",
            "value_head.linear",
            "fc2",
        ],
        "feedforward_modules": ["fc2"],
    },
    "plantcad2": {
        "target_modules": ["in_proj"],
        "feedforward_modules": [],
    },
    "agront": {  # ESM: attention.self.{key,value}, output.dense (FFN out)
        "target_modules": ["key", "value", "output.dense"],
        "feedforward_modules": ["output.dense"],
    },
    "pdllm": {  # GPT-2: fused QKV projection + MLP output (Conv1D layers)
        "target_modules": ["c_attn", "mlp.c_proj"],
        "feedforward_modules": ["mlp.c_proj"],
    },
}

CAUSAL_MODELS = {"pdllm"}


def resolve_model_id(model_key_or_id: str) -> str:
    return MODEL_REGISTRY.get(model_key_or_id, model_key_or_id)


def is_plantcad(model_key: str, model_id: str) -> bool:
    s = f"{model_key} {model_id}".lower()
    return "plantcad" in s or "caduceus" in s


def _filter_forward_kwargs(module):
    """Wrap module.forward so any kwarg its real forward does not declare is
    dropped before the call. Caduceus/Mamba declares only a small set (input_ids,
    output_hidden_states, ...) and raises on the standard transformer kwargs
    (attention_mask, output_attentions, return_dict, ...) that PEFT's
    FEATURE_EXTRACTION wrapper injects. This mirrors how the PlantCAD2 authors'
    script calls the base model with only the kwargs it accepts. Idempotent; if
    the forward accepts **kwargs, no filtering is applied."""
    if getattr(module, "_kwargs_filtered", False):
        return
    orig_forward = module.forward
    try:
        params = inspect.signature(orig_forward).parameters
        allowed = set(params)
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (ValueError, TypeError):
        allowed, has_var_kw = None, True

    def _forward(*args, **kwargs):
        if allowed is not None and not has_var_kw:
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return orig_forward(*args, **kwargs)

    module.forward = _forward
    module._kwargs_filtered = True


def autodetect_ia3_targets(model):
    """Fallback when no hint matches: every nn.Linear leaf-name suffix, with
    feedforward = common down/output-projection names."""
    linear_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_names.add(name.split(".")[-1])
    ff = [n for n in linear_names if n in {"out_proj", "output", "down_proj", "fc2"}]
    return sorted(linear_names), sorted(ff) or sorted(linear_names)[:1]


def masked_mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean over real (non-pad) tokens. hidden_states (B, L, H),
    attention_mask (B, L) -> (B, H)."""
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def last_token_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Hidden state of the last non-pad token (requires right padding).
    hidden_states (B, L, H), attention_mask (B, L) -> (B, H)."""
    last = attention_mask.sum(dim=1).long() - 1
    batch = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch, last]


class RegressionHead(nn.Module):
    """Pooled embedding -> output_dim. Softplus when non_negative (CSP/Poisson),
    else Identity (GEP regression, or logits for EAP classification)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        non_negative: bool = False,
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.activation = nn.Softplus() if non_negative else nn.Identity()

    def forward(self, x):
        return self.activation(self.net(x))


class LLMRegressor(nn.Module):
    """
    Pretrained DNA LM (NTv3 or PlantCAD2) + mean-pooled prediction head.
    forward(input_ids, attention_mask, bit=None) matches the trainer's
    self.model(**data, bit=bit) call.

    Model-specific handling, centralized:
      - PlantCAD2 (Caduceus, RC-equivariant): the final hidden state is width
        2*H, [forward | reverse-complement]; the halves are averaged (RC half
        reversed along length) to recover H, per the authors' embedding snippet.
        Caduceus accepts no attention_mask/output_attentions/etc.; its forward is
        wrapped to drop any kwarg it doesn't declare (PEFT injects several), and
        we never pass a mask ourselves.
      - NTv3 (U-Net + transformer): hidden_states[-1] is the post-deconv
        embedding at input resolution, used as-is, called WITH attention_mask.

    tuning_mode: "frozen" (head only) | "full" (all backbone) | "ia3".
    """

    def __init__(
        self,
        model_id: str,
        n_tasks: int,
        tuning_mode: str = "frozen",
        head_hidden_dim: Optional[int] = None,
        head_dropout: float = 0.1,
        load_backbone_weights: bool = True,
        non_negative_head: bool = False,
        ia3_target_modules: Optional[list] = None,
        ia3_feedforward_modules: Optional[list] = None,
    ):
        super().__init__()
        self._model_key = model_id
        self.model_id = resolve_model_id(model_id)
        self.is_plantcad = is_plantcad(self._model_key, self.model_id)

        # ---- load backbone ----
        fallback = (
            AutoModelForCausalLM
            if self._model_key in CAUSAL_MODELS
            else AutoModelForMaskedLM
        )
        if load_backbone_weights:
            try:
                self.backbone = AutoModel.from_pretrained(
                    self.model_id, trust_remote_code=True
                )
                self._use_mlm = False
            except (ValueError, KeyError):
                self.backbone = fallback.from_pretrained(
                    self.model_id, trust_remote_code=True
                )
                self._use_mlm = True
        else:
            cfg = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            try:
                self.backbone = AutoModel.from_config(cfg, trust_remote_code=True)
                self._use_mlm = False
            except (ValueError, TypeError):
                self.backbone = fallback.from_config(cfg, trust_remote_code=True)
                self._use_mlm = True

        # PlantCAD2/Caduceus: make the raw backbone drop kwargs it doesn't accept
        # before any PEFT wrapping.
        if self.is_plantcad:
            _filter_forward_kwargs(self.backbone)
        # Some DNA tokenizers have more tokens than the model's embedding table
        # (mirrors the PDLLM authors' fine-tuning script).
        if self._model_key in CAUSAL_MODELS:
            tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            if self.backbone.config.vocab_size < len(tok):
                self.backbone.resize_token_embeddings(len(tok))
            if tok.pad_token_id is not None:
                self.backbone.config.pad_token_id = tok.pad_token_id
        # ---- tuning mode ----
        self.tuning_mode = tuning_mode
        self.freeze_backbone = tuning_mode in ("frozen", "ia3")

        if tuning_mode == "frozen":
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        elif tuning_mode == "ia3":
            key = self._model_key if self._model_key in IA3_TARGET_HINTS else None
            if ia3_target_modules is not None:
                tgt = ia3_target_modules
                ff = ia3_feedforward_modules or []
            elif key:
                tgt = IA3_TARGET_HINTS[key]["target_modules"]
                ff = IA3_TARGET_HINTS[key]["feedforward_modules"]
            else:
                tgt, ff = autodetect_ia3_targets(self.backbone)
            ia3_cfg = IA3Config(
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=tgt,
                feedforward_modules=ff,
            )
            self.backbone = get_peft_model(self.backbone, ia3_cfg)
            if hasattr(self.backbone, "print_trainable_parameters"):
                self.backbone.print_trainable_parameters()
            # Re-filter the wrapped base's forward — the one PEFT actually calls
            # (tuners_utils forwards *args, **kwargs straight into it).
            if self.is_plantcad:
                inner = getattr(self.backbone, "base_model", self.backbone)
                inner = getattr(inner, "model", inner)
                _filter_forward_kwargs(inner)
        # "full" -> all backbone params stay trainable

        # ---- head (embed dim inferred from a real forward, post RC-fold) ----
        self.embed_dim = self._infer_embed_dim()
        self.head = RegressionHead(
            input_dim=self.embed_dim,
            output_dim=n_tasks,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            non_negative=non_negative_head,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.tuning_mode in ("frozen", "ia3"):
            self.backbone.eval()
        return self

    def _infer_embed_dim(self) -> int:
        was_training = self.backbone.training
        self.backbone.eval()

        # Caduceus fused kernels (Triton) require CUDA; a CPU probe crashes.
        # Probe on the backbone's device, promoting to CUDA if it's on CPU and
        # CUDA is available.
        dev = next(self.backbone.parameters()).device
        if dev.type == "cpu" and torch.cuda.is_available():
            self.backbone.to("cuda")
            dev = torch.device("cuda")

        with torch.no_grad():
            dummy_ids = torch.zeros((1, 128), dtype=torch.long, device=dev)
            dummy_mask = torch.ones((1, 128), dtype=torch.long, device=dev)
            hs = self._backbone_hidden_states(dummy_ids, dummy_mask)

        if was_training and not self.freeze_backbone:
            self.backbone.train()
        return hs.shape[-1]

    def _backbone_hidden_states(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Final-layer token embeddings (B, L, H), with PlantCAD2 RC-folding."""
        call_kwargs = dict(input_ids=input_ids, output_hidden_states=True)
        if not self.is_plantcad:
            call_kwargs["attention_mask"] = attention_mask
        outputs = self.backbone(**call_kwargs)

        hs = getattr(outputs, "hidden_states", None)
        hs = hs[-1] if hs is not None else outputs.last_hidden_state

        if self.is_plantcad:
            # RC-equivariant output: [forward | reverse-complement], width 2*H.
            H = hs.shape[-1] // 2
            forward = hs[..., :H]
            reverse = hs[..., H:].flip(dims=[1])  # reverse along length
            hs = (forward + reverse) / 2.0
        return hs

    def forward(self, input_ids, attention_mask, bit=None):
        if self.tuning_mode == "frozen":
            with torch.no_grad():
                hs = self._backbone_hidden_states(input_ids, attention_mask)
        else:
            hs = self._backbone_hidden_states(input_ids, attention_mask)
        if self._model_key in CAUSAL_MODELS:
            pooled = last_token_pool(hs, attention_mask)
        else:
            pooled = masked_mean_pool(hs, attention_mask)
        return self.head(pooled)


def build_model(
    args: Any,
    new_model: Optional[bool] = True,
    model_path: Optional[str] = None,
):
    """
    Reads from args (LLMConfig): model_name, n_tasks, tuning_mode,
    head_hidden_dim, head_dropout, task, loss, ia3_target_modules,
    ia3_feedforward_modules.

    Head activation: Softplus for CSP or a Poisson loss (non-negative targets),
    Identity otherwise (GEP regression / EAP logits).

    On resume, strips a DDP "module." prefix and drops length-dependent RoPE
    caches (recomputed on first forward), then verifies nothing else is missing.
    """
    tuning_mode = getattr(args, "tuning_mode", None)
    if tuning_mode is None:
        tuning_mode = "frozen" if getattr(args, "freeze_backbone", True) else "full"

    task = getattr(args, "task", "CSP")
    non_negative_head = (task.upper() == "CSP") or (args.loss.upper() == "POISSON")
    if task.upper() == "EAP":
        non_negative_head = False  # classification logits, never non-negative

    model = LLMRegressor(
        model_id=args.model_name,
        n_tasks=args.n_tasks,
        tuning_mode=tuning_mode,
        head_hidden_dim=getattr(args, "head_hidden_dim", None),
        head_dropout=getattr(args, "head_dropout", 0.1),
        load_backbone_weights=True,
        non_negative_head=non_negative_head,
        ia3_target_modules=getattr(args, "ia3_target_modules", None),
        ia3_feedforward_modules=getattr(args, "ia3_feedforward_modules", None),
    )

    if not new_model and model_path is not None:
        state = torch.load(model_path, map_location="cpu", weights_only=True)

        if any(k.startswith("module.") for k in state):
            state = {
                (k[len("module.") :] if k.startswith("module.") else k): v
                for k, v in state.items()
            }
        state = {
            k: v
            for k, v in state.items()
            if not (
                k.endswith("rotary_embedding.cos_cached")
                or k.endswith("rotary_embedding.sin_cached")
            )
        }
        result = model.load_state_dict(state, strict=False)
        leftover = [
            k
            for k in (list(result.missing_keys) + list(result.unexpected_keys))
            if "rotary_embedding" not in k
        ]
        if leftover:
            raise RuntimeError(
                f"Unexpected missing/unexpected keys on checkpoint load: {leftover}"
            )

    return model
