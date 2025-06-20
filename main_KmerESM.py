import argparse
import os
import seaborn as sns

sns.set_theme()
import torch
from src.utils import (
    create_path,
    save_data_to_csv,
    generate_UDir,
    read_json,
    get_device,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from src.tokenizers import KmerEsmTokenizer
import torch.multiprocessing as mp
from torch.cuda import device_count
from src.dataset_utils import load_dataset
from src.train_utils import trainer
from src.results_utils import evaluate_model

from src.DeepPlant_kmers import build_model

from src.seed import set_seed
from src.config import DeepPlantKmerConfig
from src.optimizers import ScheduledOptim
from src.losses import CustomCosineEmbeddingLoss
from src.logger import configure_logging_format
from typing import Optional, Any
from src.ddp import setup, cleanup, is_main_process
import torch.distributed as dist


set_seed()


def parse_arguments(parser):
    parser.add_argument("--json", type=str, help="path to the json file")
    parser.add_argument(
        "-n",
        "--new",
        action="store_true",
        help="Build a new logistic regression model",
    )
    parser.add_argument("-m", "--model", type=str, help="Existing model path")
    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Train the model",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        action="store_true",
        help="Evaluate the model",
    )
    args = parser.parse_args()
    return args


def main(
    device: Any,
    n_gpu: Optional[int] = 0,
    new_model: Optional[bool] = True,
    model_path: Optional[str] = None,
    config: Optional[str] = None,
    model_folder_path: Optional[str] = "",
    train: Optional[bool] = True,
    evaluate: Optional[bool] = True,
    data_class: Optional[Any] = None,
):
    logger = configure_logging_format(file_path=model_folder_path)
    model = build_model(args=config, new_model=new_model, model_path=model_path).to(
        device=device
    )
    tokenizer = KmerEsmTokenizer.from_pretrained(
        config.tokenizer_path,
    )
    if n_gpu > 1:

        setup(device, n_gpu)

        model = DDP(
            model,
            device_ids=[device],
            find_unused_parameters=config.find_unused_parameters,
        )
        if new_model:
            if is_main_process():
                with open(
                    file=os.path.join(model_folder_path, "model.txt"), mode="w"
                ) as f:
                    print(
                        sum(p.numel() for p in model.parameters() if p.requires_grad),
                        file=f,
                    )
                    print(model, file=f)
                torch.save(
                    model.state_dict(),
                    os.path.join(model_folder_path, "temp_checkpoint.pt"),
                )
            dist.barrier()
            map_location = {
                "cuda:%d" % 0: "cuda:%d" % device,
                "cpu": "cuda:%d" % device,
            }
            model.load_state_dict(
                torch.load(
                    os.path.join(model_folder_path, "temp_checkpoint.pt"),
                    map_location=map_location,
                    weights_only=True,
                )
            )
            dist.barrier()
            if is_main_process():
                os.remove(os.path.join(model_folder_path, "temp_checkpoint.pt"))

    # prepare the optimizer
    optimizer = ScheduledOptim(config)
    optimizer(model.parameters())

    # Prepare the loss function
    loss_function = CustomCosineEmbeddingLoss(config)
    # loss_function = torch.nn.MSELoss(reduction="mean")

    if train:
        # Prepare the data
        logger.info(f"Device: {device} - Loading train dataset")
        train_loader = data_class.get_dataloader(
            indices_paths=config.train_indices_path,
            device=device,
            n_gpu=n_gpu,
            tokenizer=tokenizer,
            **config.dict(),
        )
        logger.info(f"Device: {device} - Loading valid dataset")
        valid_loader = data_class.get_dataloader(
            indices_paths=config.valid_indices_path,
            device=device,
            n_gpu=n_gpu,
            tokenizer=tokenizer,
            **config.dict(),
        )
        # Train model
        trainer_ = trainer(
            model=model,
            loss_fn=loss_function,
            device=device,
            train_dataloader=train_loader,
            valid_dataloader=valid_loader,
            model_folder_path=model_folder_path,
            optimizer=optimizer,
            **config.dict(),
        )
        best_model, model_path = trainer_.train()

        if n_gpu > 1:
            dist.barrier()
            if not is_main_process():
                 map_location = {
                    "cuda:%d" % 0: "cuda:%d" % device,
                    "cpu": "cuda:%d" % device,
                }
                model.load_state_dict(
                    torch.load(
                        model_path,
                        map_location=map_location,
                        weights_only=True,
                    )
                )

    else:
        best_model = model

    if evaluate:
        if n_gpu > 1:
            dist.barrier()
        logger.info(f"Device: {device} - Loading test dataset")
        test_loader = data_class.get_dataloader(
            indices_paths=config.test_indices_path,
            device=device,
            n_gpu=n_gpu,
            tokenizer=tokenizer,
            augment_data=False,
            **config.dict(),
        )
        # Evaluate model
        mse, pearson, spearman = evaluate_model(
            model=best_model,
            dataloader=test_loader,
            device=device,
            model_folder_path=model_folder_path,
            experiment_names=config.experiment_name,
        )
        data_dict = {
            "path": model_path,
            "mse": mse,
            "pearson": pearson,
            "spearman": spearman,
        }
        results_csv_path = os.path.join(config.results_path, "results.csv")
        # Save model performance
        if is_main_process():
            save_data_to_csv(data_dictionary=data_dict, csv_file_path=results_csv_path)
    if n_gpu > 1:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate LR model")
    args = parse_arguments(parser)
    assert (
        args.json != None
    ), "Please specify the path to the json file with --json json_path"
    assert os.path.exists(
        args.json
    ), f"The path to the json file {args.json} does not exist. Please verify"
    assert (args.new == True) ^ (
        (args.model) != None
    ), "Wrong arguments. Either include -n to build a new model or specify -m model_path"

    config = DeepPlantKmerConfig(**read_json(json_path=args.json))
    config_dict = config.dict()
    device = get_device()

    # prepare the model
    if args.new:
        Udir = generate_UDir(path=config.results_path)
        model_folder_path = os.path.join(config.results_path, Udir)
        create_path(model_folder_path)
    else:
        model_folder_path = os.path.dirname(args.model)
        model_path = args.model

    print(f"Model path is {model_folder_path}")
    logger = configure_logging_format(file_path=model_folder_path)
    for key, value in config_dict.items():
        logger.info(f"Device: {device} - {key}: {value}")
    logger.info(f"Device: {device} - Processing data files")
    data_class = load_dataset(
        sequences_paths=config.sequences_paths,
        labels_paths=config.labels_paths,
    )

    if device == "cuda":
        n_gpu = device_count()
        logger.info(f"Using {n_gpu} gpu(s)")
        mp.spawn(
            main,
            args=(
                n_gpu,
                args.new,
                args.model,
                config,
                model_folder_path,
                args.train,
                args.evaluate,
                data_class,
            ),
            nprocs=n_gpu,
            join=True,
        )
    else:
        logger.info(f"Using cpu")
        main(
            device=device,
            n_gpu=0,
            new_model=args.new,
            model_path=args.model,
            config=config,
            model_folder_path=model_folder_path,
            train=args.train,
            evaluate=args.evaluate,
            data_class=data_class,
        )
