from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class CSPConfig:
    h5_paths: list = field(default_factory=lambda: [""])
    train_indices_path: list = field(default_factory=lambda: [""])
    valid_indices_path: list = field(default_factory=lambda: [""])
    test_indices_path: list = field(default_factory=lambda: [""])
    experiment_name: list = field(default_factory=lambda: [""])
    sequence_length: int = 2500
    results_path: str = "results/results_DeepPlant"
    optimizer_type: str = ("sgd",)
    weight_decay: float = (0.001,)
    momentum: float = (0.95,)
    use_scheduler: bool = (False,)
    max_epochs: int = (30,)
    warmup_steps: int = (30,)
    warmup_begin_lr: float = (0.0001,)
    max_lr: float = (0.01,)
    final_lr: float = (0.0001,)
    loss: str = ("mse",)
    dist: str = "cosine"
    metric: str = "pearson"
    alpha: float = (0.1,)
    lazyLoad: bool = (False,)
    batchSize: int = (512,)
    num_workers: int = (0,)
    counter_for_early_stop_threshold: int = (5,)
    epochs_to_check_loss: int = (1,)
    n_accumulated_batches: int = (1,)
    n_filters: int = (240,)
    encoder_type: str = "attention"
    embed_dim: int = 2048
    num_heads: int = 8
    dim_forwardfeed: int = 4096
    dropout: float = 0.2
    encoder_num_layers: int = 1
    decoder_num_layers: int = 0
    expand_factor: int = 4
    n_features: list = field(default_factory=lambda: [2837])
    find_unused_parameters: bool = False
    use_pos_encoding: bool = True
    max_seq_length: int = 100
    consistency_regularization: bool = False

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class GEPConfig:
    h5_paths: list = field(default_factory=lambda: [""])
    train_indices_path: list = field(default_factory=lambda: [""])
    valid_indices_path: list = field(default_factory=lambda: [""])
    test_indices_path: list = field(default_factory=lambda: [""])
    experiment_name: list = field(default_factory=lambda: [""])
    sequence_length: int = 2500
    results_path: str = "results/results_DeepPlant"
    task: str = "GEP"
    freeze_mode: str = "none"
    optimizer_type: str = ("sgd",)
    weight_decay: float = (0.001,)
    momentum: float = (0.95,)
    use_scheduler: bool = (False,)
    max_epochs: int = (30,)
    warmup_steps: int = (30,)
    warmup_begin_lr: float = (0.0001,)
    max_lr: float = (0.01,)
    final_lr: float = (0.0001,)
    loss: str = ("mse",)
    metric: str = "pearson"
    lazyLoad: bool = (False,)
    batchSize: int = (512,)
    num_workers: int = (0,)
    counter_for_early_stop_threshold: int = (5,)
    epochs_to_check_loss: int = (1,)
    n_accumulated_batches: int = (1,)
    n_filters: int = (240,)
    encoder_type: str = "attention"
    embed_dim: int = 2048
    num_heads: int = 8
    dim_forwardfeed: int = 4096
    dropout: float = 0.2
    encoder_num_layers: int = 1
    decoder_num_layers: int = 0
    expand_factor: int = 4
    n_features: list = field(default_factory=lambda: [2837])
    find_unused_parameters: bool = False
    use_pos_encoding: bool = True
    max_seq_length: int = 100
    consistency_regularization: bool = False

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class EAPConfig:
    sequences_paths: list = field(default_factory=lambda: [""])
    results_path: str = "results/results_DeepPlant"
    targets: str = "ALL"
    task: str = "EAP"
    use_reverse_complement: bool = False
    freeze_mode: str = "none"
    optimizer_type: str = ("sgd",)
    weight_decay: float = (0.001,)
    momentum: float = (0.95,)
    use_scheduler: bool = (False,)
    max_epochs: int = (30,)
    warmup_steps: int = (30,)
    warmup_begin_lr: float = (0.0001,)
    max_lr: float = (0.01,)
    final_lr: float = (0.0001,)
    l1_lambda: float = 0.01
    lazyLoad: bool = (False,)
    loss: str = "ce"
    focal_gamma: float = 2.0
    focal_alpha: Optional[list] = None  # per-class weights [non-enh, enh]
    class_weights: Optional[list] = None  # for "wce"
    metric: str = "auprc"
    batchSize: int = (512,)
    num_workers: int = (0,)
    counter_for_early_stop_threshold: int = (5,)
    epochs_to_check_loss: int = (1,)
    n_accumulated_batches: int = (1,)
    n_filters: int = (240,)
    encoder_type: str = "attention"
    embed_dim: int = 2048
    num_heads: int = 8
    dim_forwardfeed: int = 4096
    dropout: float = 0.2
    encoder_num_layers: int = 1
    decoder_num_layers: int = 0
    expand_factor: int = 4
    n_features: list = field(default_factory=lambda: [2837])
    find_unused_parameters: bool = False
    use_pos_encoding: bool = True
    max_seq_length: int = 100
    consistency_regularization: bool = False
    use_imbalanced_loader: bool = False

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class LLMConfig:
    # ---- data ----
    h5_paths: list = field(default_factory=lambda: [""])
    train_indices_path: list = field(default_factory=lambda: [""])
    valid_indices_path: list = field(default_factory=lambda: [""])
    test_indices_path: list = field(default_factory=lambda: [""])
    experiment_name: list = field(default_factory=lambda: [""])
    sequence_length: int = 2500
    results_path: str = "results/results_LLM"
    sequences_paths: list = field(default_factory=lambda: [""])
    use_reverse_complement: bool = False
    targets: str = "ALL"

    # ---- LLM backbone + head ----
    model_name: str = "nt"  # "nt" | "plantcad2" | raw HF id
    task: str = "CSP"

    n_tasks: int = 2835  # number of regression targets (label width)
    freeze_backbone: bool = True  # True = linear probe, False = full fine-tune
    head_hidden_dim: int = None  # None -> defaults to embedding width
    head_dropout: float = 0.1
    max_length: int = None  # tokens; ~1000 for NT, ~6000 for PlantCAD2
    tuning_mode: str = "frozen"  # "frozen" | "full" | "ia3"
    ia3_target_modules: list = None  # None -> per-model hint or autodetect
    ia3_feedforward_modules: list = None
    # ---- optimizer ----
    optimizer_type: str = "adamw"
    weight_decay: float = 0.01
    momentum: float = 0.95
    use_scheduler: bool = False
    max_epochs: int = 100
    warmup_steps: int = 5
    warmup_begin_lr: float = 0.0001
    max_lr: float = 0.0005
    final_lr: float = 0.0005

    # ---- loss / metric ----
    loss: str = "mse"  # mse | huber | rmse | poisson
    metric: str = "pearson"  # pearson | loss

    # ---- training loop ----
    lazyLoad: bool = True
    batchSize: int = 16
    num_workers: int = 8
    counter_for_early_stop_threshold: int = 10
    epochs_to_check_loss: int = 1
    n_accumulated_batches: int = 4

    # ---- DDP ----
    find_unused_parameters: bool = False

    # consistency regularization is DeepPlant-only; always off for the LLM path
    consistency_regularization: bool = False

    def dict(self):
        return {k: v for k, v in asdict(self).items()}
