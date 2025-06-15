from dataclasses import dataclass, asdict, field


@dataclass
class DeepPlantConfig:
    optimizer_type: str = ("sgd",)
    use_scheduler: bool = (False,)
    warmup_steps: int = (30,)
    warmup_begin_lr: float = (0.0001,)
    max_lr: float = (0.01,)
    final_lr: float = (0.0001,)
    weight_decay: float = (0.001,)
    momentum: float = (0.95,)
    lazyLoad: bool = (False,)
    loss: str = ("mse",)
    alpha: float = (0.1,)
    gamma: float = (0.1,)
    fraction: float = (0.9,)
    max_epochs: int = (30,)
    batchSize: int = (512,)
    num_workers: int = (0,)
    counter_for_early_stop_threshold: int = (5,)
    epochs_to_check_loss: int = (1,)
    n_accumulated_batches: int = (1,)
    results_path: str = "results/results_DeepPlant"
    sequences_path: list = field(default_factory=lambda: [""])
    labels_path: list = field(default_factory=lambda: [""])
    train_indices_path: list = field(default_factory=lambda: [""])
    valid_indices_path: list = field(default_factory=lambda: [""])
    test_indices_path: list = field(default_factory=lambda: [""])
    experiment_name: list = field(default_factory=lambda: [""])
    models_list: list = field(default_factory=lambda: [""])
    bits: list = field(default_factory=lambda: [0])
    use_reverse_complement: bool = False
    n_filters: int = (240,)
    backbone_path: str = ""
    load_pretrain: bool = True
    freeze_pretrain: bool = False
    embed_dim: int = 2048
    num_heads: int = 8
    dim_forwardfeed: int = 4096
    dropout: float = 0.2
    encoder_num_layers: int = 1
    decoder_num_layers: int = 0
    n_features: list = field(default_factory=lambda: [2837])
    find_unused_parameters: bool = False
    use_pretrained_filter: bool = False
    n_label_embeddings: int = 0
    encoder_type: str = "mamba"
    use_pos_encoding: bool = True
    max_seq_length: int = 100

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class DeepPlantKmerConfig:
    optimizer_type: str = ("sgd",)
    use_scheduler: bool = (False,)
    warmup_steps: int = (30,)
    warmup_begin_lr: float = (0.0001,)
    max_lr: float = (0.01,)
    final_lr: float = (0.0001,)
    weight_decay: float = (0.001,)
    momentum: float = (0.95,)
    lazyLoad: bool = (False,)
    loss: str = ("mse",)
    dist: str = "cosine"
    alpha: float = (0.1,)
    gamma: float = (0.1,)
    fraction: float = (0.9,)
    max_epochs: int = (30,)
    batchSize: int = (32,)
    num_workers: int = (0,)
    counter_for_early_stop_threshold: int = (5,)
    epochs_to_check_loss: int = (1,)
    n_accumulated_batches: int = (1,)
    results_path: str = "results/results_DeepPlant"
    sequences_paths: list = field(default_factory=lambda: [""])
    labels_paths: list = field(default_factory=lambda: [""])
    train_indices_path: list = field(default_factory=lambda: [""])
    valid_indices_path: list = field(default_factory=lambda: [""])
    test_indices_path: list = field(default_factory=lambda: [""])
    experiment_name: list = field(default_factory=lambda: [""])
    models_list: list = field(default_factory=lambda: [""])
    bits: list = field(default_factory=lambda: [0])
    use_reverse_complement: bool = False
    n_filters: int = (240,)
    backbone_path: str = ""
    load_pretrain: bool = True
    freeze_pretrain: bool = False
    embed_dim: int = 2048
    num_heads: int = 8
    dim_forwardfeed: int = 4096
    dropout: float = 0.2
    encoder_num_layers: int = 1
    decoder_num_layers: int = 0
    n_features: list = field(default_factory=lambda: [2837])
    find_unused_parameters: bool = False
    use_pretrained_filter: bool = False
    n_label_embeddings: int = 0
    encoder_type: str = "mamba"
    tokenizer_path: str = ""
    vocab_size: int = 1030
    use_pos_encoding: bool = True
    add_special_tokens: bool = True
    max_seq_length: int = 512

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class ExpressionConfig:
    optimizer_type: str = ("sgd",)
    use_scheduler: bool = (False,)
    warmup_steps: int = (30,)
    warmup_begin_lr: float = (0.0001,)
    max_lr: float = (0.01,)
    final_lr: float = (0.0001,)
    weight_decay: float = (0.001,)
    momentum: float = (0.95,)
    lazy_loading: bool = (False,)
    loss: str = ("mse",)
    alpha: float = (0.1,)
    gamma: float = (0.1,)
    fraction: float = (0.9,)
    max_epochs: int = (30,)
    batch_size: int = (512,)
    num_workers: int = (0,)
    counter_for_early_stop_threshold: int = (5,)
    epochs_to_check_loss: int = (1,)
    n_accumulated_batches: int = (1,)
    results_path: str = "results/results_DeepPlant"
    sequences_path: list = field(default_factory=lambda: [""])
    labels_path: list = field(default_factory=lambda: [""])
    train_indices_path: list = field(default_factory=lambda: [""])
    valid_indices_path: list = field(default_factory=lambda: [""])
    test_indices_path: list = field(default_factory=lambda: [""])
    experiment_name: list = field(default_factory=lambda: [""])
    models_list: list = field(default_factory=lambda: [""])
    bits: list = field(default_factory=lambda: [0])
    use_reverse_complement: bool = False
    n_filters: int = (240,)
    backbone_path: str = ""
    load_pretrain: bool = True
    freeze_pretrain: bool = False
    embed_dim: int = 2048
    num_heads: int = 8
    dim_forwardfeed: int = 4096
    dropout: float = 0.2
    encoder_num_layers: int = 1
    decoder_num_layers: int = 0
    n_features: list = field(default_factory=lambda: [2837])
    find_unused_parameters: bool = False
    use_pretrained_filter: bool = False
    encoder_type: str = "mamba"
    use_pos_encoding: bool = True
    max_seq_length: int = 100

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class EnhancerConfig:
    optimizer_type: str = ("sgd",)
    use_scheduler: bool = (False,)
    warmup_steps: int = (30,)
    warmup_begin_lr: float = (0.0001,)
    max_lr: float = (0.01,)
    final_lr: float = (0.0001,)
    weight_decay: float = (0.001,)
    momentum: float = (0.95,)
    lazy_loading: bool = (False,)
    max_epochs: int = (30,)
    batch_size: int = (512,)
    num_workers: int = (0,)
    counter_for_early_stop_threshold: int = (5,)
    epochs_to_check_loss: int = (1,)
    n_accumulated_batches: int = (1,)
    results_path: str = "results/results_DeepPlant"
    sequences_paths: list = field(default_factory=lambda: [""])
    experiment_name: list = field(default_factory=lambda: [""])
    bits: list = field(default_factory=lambda: [0])
    use_reverse_complement: bool = False
    n_filters: int = (240,)
    load_pretrain: bool = True
    freeze_pretrain: bool = False
    embed_dim: int = 2048
    num_heads: int = 8
    dim_forwardfeed: int = 4096
    dropout: float = 0.2
    encoder_num_layers: int = 1
    decoder_num_layers: int = 0
    n_features: list = field(default_factory=lambda: [2837])
    find_unused_parameters: bool = False
    use_pretrained_filter: bool = False
    encoder_type: str = "mamba"
    use_pos_encoding: bool = True
    max_seq_length: int = 100

    def dict(self):
        return {k: v for k, v in asdict(self).items()}
