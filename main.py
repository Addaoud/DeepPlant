import argparse
import os
import seaborn as sns

sns.set_theme()
import torch
from src.utils import (
    create_path,
    save_model_log,
    save_data_to_csv,
    generate_UDir,
    read_json,
    get_device,
)
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.cuda import device_count
from src.dataset_utils import load_dataset
from src.train_utils import trainer
from src.results_utils import evaluate_model

from src.DeepPlant_simple import build_model

# from src.sei import build_model

from src.seed import set_seed
from src.config import DeepPlantConfig
from src.optimizers import ScheduledOptim
from src.losses import PoissonNLLLoss, MSE
from typing import Optional, Any
from src.ddp import setup, cleanup, is_main_process
import torch.distributed as dist

# import logging

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
    model = build_model(args=config, new_model=new_model, model_path=model_path).to(
        device=device
    )
    print(f"Model path is {model_folder_path}")
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
                    print(model, file=f)
                torch.save(
                    model.state_dict(),
                    os.path.join(model_folder_path, "temp_checkpoint.pt"),
                )
            dist.barrier()
            map_location = {"cuda:%d" % 0: "cuda:%d" % device}
            model.load_state_dict(
                torch.load(
                    os.path.join(model_folder_path, "temp_checkpoint.pt"),
                    map_location=map_location,
                    weights_only=True,
                )
            )

            if is_main_process():
                os.remove(os.path.join(model_folder_path, "temp_checkpoint.pt"))

    # prepare the optimizer
    optimizer = ScheduledOptim(config)
    optimizer(model.parameters())

    # Prepare the loss function
    loss_function = PoissonNLLLoss(config)
    # loss_function = torch.nn.MSELoss(reduction="mean")

    if train:
        # Prepare the data
        print(f"Loading train data on device {device}")
        train_loader = data_class.get_dataloader(
            indices_paths=config.train_indices_path,
            batchSize=config.batch_size,
            lazyLoad=config.lazy_loading,
            device=device,
            n_gpu=n_gpu,
            num_workers=config.num_workers,
        )
        print(f"Loading valid data on device {device}")
        valid_loader = data_class.get_dataloader(
            indices_paths=config.valid_indices_path,
            batchSize=config.batch_size,
            lazyLoad=config.lazy_loading,
            device=device,
            n_gpu=n_gpu,
            num_workers=config.num_workers,
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
        if is_main_process():
            save_model_log(log_dir=model_folder_path, data_dictionary={})

        if n_gpu > 1:
            dist.barrier()
            if not is_main_process():
                map_location = {"cuda:%d" % 0: "cuda:%d" % device}
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
        print(f"Loading test data on device {device}")
        test_loader = data_class.get_dataloader(
            indices_paths=config.test_indices_path,
            batchSize=config.batch_size,
            lazyLoad=config.lazy_loading,
            device=device,
            n_gpu=n_gpu,
            num_workers=config.num_workers,
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

    config = DeepPlantConfig(**read_json(json_path=args.json))
    config_dict = config.dict()
    device = get_device()
    config_dict["device"] = device

    # prepare the model
    if args.new:
        Udir = generate_UDir(path=config.results_path)
        model_folder_path = os.path.join(config.results_path, Udir)
        create_path(model_folder_path)
        save_model_log(log_dir=model_folder_path, data_dictionary=config_dict)
    else:
        model_folder_path = os.path.dirname(args.model)
        model_path = args.model

    data_class = load_dataset(
        sequences_paths=config.sequences_path,
        labels_paths=config.labels_path,
    )

    if device == "cuda":
        n_gpu = device_count()
        print(f"Using {n_gpu} gpu(s)")
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
