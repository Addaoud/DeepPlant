from dataclasses import dataclass, asdict, field


@dataclass
class CSPConfig:
    h5_paths: list = field(default_factory=lambda: [""])
    train_indices_path: list = field(default_factory=lambda: [""])
    valid_indices_path: list = field(default_factory=lambda: [""])
    test_indices_path: list = field(default_factory=lambda: [""])
    experiment_name: list = field(default_factory=lambda: [""])
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
class ExpressionConfig:
    h5_paths: list = field(default_factory=lambda: [""])
    train_indices_path: list = field(default_factory=lambda: [""])
    valid_indices_path: list = field(default_factory=lambda: [""])
    test_indices_path: list = field(default_factory=lambda: [""])
    experiment_name: list = field(default_factory=lambda: [""])
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

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class EnhancerConfig:
    sequences_paths: list = field(default_factory=lambda: [""])
    results_path: str = "results/results_DeepPlant"
    use_reverse_complement: bool = False
    optimizer_type: str = ("sgd",)
    weight_decay: float = (0.001,)
    momentum: float = (0.95,)
    use_scheduler: bool = (False,)
    max_epochs: int = (30,)
    warmup_steps: int = (30,)
    warmup_begin_lr: float = (0.0001,)
    max_lr: float = (0.01,)
    final_lr: float = (0.0001,)
    lazyLoad: bool = (False,)
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

    def dict(self):
        return {k: v for k, v in asdict(self).items()}
