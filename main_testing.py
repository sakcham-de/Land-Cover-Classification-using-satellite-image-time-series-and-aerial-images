import argparse
import os
from pathlib import Path 

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy

try:
    from pytorch_lightning.utilities.distributed import rank_zero_only
except ImportError:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

import albumentations as A

from src.backbones.txt_model import TimeTexture_flair
from src.datamodule import DataModule
from src.task_module import SegmentationTask
from src.utils_dataset import read_config
from src.load_data import load_data
from src.prediction_writer import PredictionWriter
from src.metrics import generate_miou, generate_mf1s

from src.utils_prints import (
    print_config,
    print_recap,
    print_metrics,
    print_inference_time, 
    print_iou_metrics, 
    print_f1_metrics, 
    print_overall_accuracy
)


argParser = argparse.ArgumentParser()
argParser.add_argument("--config_file", help="Path to the .yml config file")


def main(config):

    seed_everything(2022, workers=True)
    out_dir = Path(config["outputs"]["out_folder"], config["outputs"]["out_model_name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    d_train, d_val, d_test = load_data(config)
   

    # Augmentation
    if config["use_augmentation"] == True:
        transform_set = A.Compose([A.VerticalFlip(p=0.5),
                                   A.HorizontalFlip(p=0.5),
                                   A.RandomRotate90(p=0.5),
                                #   A.RandomSnow(p=0.3),
                                   A.RandomBrightnessContrast(p=0.4),
                                   A.RandomGamma(p=0.4),
                                   A.GaussianBlur(p=0.3)])
    else:
        transform_set = None   
    
    # Dataset definition
    data_module = DataModule(
        dict_train=d_train,
        dict_val=d_val,
        dict_test=d_test,
        config=config,
        drop_last=True,
        augmentation_set = transform_set 
    )

    model = TimeTexture_flair(config)
    # model = model.load_from_checkpoint("/home/nhgnps01/flair2/flair-2-baseline/checkpoints/ckpt-epoch=27-val_loss=1.87_flair-2-baseline.ckpt")
    # checkpoint = torch.load("/home/nhgnps01/flair2/flair-2-baseline/checkpoints/ckpt-epoch=15-val_loss=1.91_flair-2-baseline.ckpt")
    # model.load_state_dict(checkpoint['state_dict'])

    #@rank_zero_only
    #def track_model():
    #    print(model)
    #track_model()

    # Optimizer and Loss
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])

    with torch.no_grad():
        weights_aer = torch.FloatTensor(np.array(list(config['weights_aerial_satellite'].values()))[:,0])
        weights_sat = torch.FloatTensor(np.array(list(config['weights_aerial_satellite'].values()))[:,1])
    criterion_vhr = nn.CrossEntropyLoss(weight=weights_aer)
    criterion_hr = nn.CrossEntropyLoss(weight=weights_sat)
    
    seg_module = SegmentationTask(
        model=model,
        num_classes=config["num_classes"],
        criterion=nn.ModuleList([criterion_vhr, criterion_hr]),
        optimizer=optimizer,
        config=config
    )
    # seg_module = model.load_state_dict(torch.load("/home/nhgnps01/flair2/flair-2-baseline/checkpoints/ckpt-epoch=15-val_loss=1.91_flair-2-baseline.ckpt")['state_dict'])

    # Callbacks

    ckpt_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(out_dir,"checkpoints"),
        filename="ckpt-{epoch:02d}-{val_loss:.2f}"+'_'+config["outputs"]["out_model_name"],
        save_top_k=1,
        mode="min",
        save_last=True
        # save_weights_only=True, # can be changed accordingly
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=30, # if no improvement after 30 epoch, stop learning. 
        mode="min",
    )

    prog_rate = TQDMProgressBar(refresh_rate=config["progress_rate"])

    callbacks = [
        ckpt_callback, 
        early_stop_callback,
        prog_rate,
    ]

    #Logger

    logger = TensorBoardLogger(
        save_dir=out_dir,
        name=Path("tensorboard_logs"+'_'+config["outputs"]["out_model_name"]).as_posix()
    )

    loggers = [
        logger
    ]

    # Train 
    trainer = Trainer(
        detect_anomaly=True,
        accelerator=config["accelerator"],
        devices=config["gpus_per_node"],
        strategy=config["strategy"],
        # strategy=DDPStrategy(find_unused_parameters=True),
        num_nodes=config["num_nodes"],
        max_epochs=config["num_epochs"],
        log_every_n_steps=100,  # 50 default used by the Trainer
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=config["enable_progress_bar"],
    )

    # if config["resume"]:
    #     ckpt_path = Path(config["outputs"]["out_folder"], config["outputs"]["out_model_name"], "checkpoints/last.ckpt")
    #     trainer.fit(seg_module, datamodule=data_module, ckpt_path=ckpt_path)
    # else:
    #     trainer.fit(seg_module, datamodule=data_module)

    # # Check metrics on validation set

    # trainer.validate(seg_module, datamodule=data_module)

    # Predict
    writer_callback = PredictionWriter(
        output_dir=os.path.join(
            out_dir, "predictions" + "_" + config["outputs"]["out_model_name"]
        ),
        write_interval="batch",
    )

    # Predict Trainer
    trainer = Trainer(
        accelerator=config["accelerator"],
        devices=config["gpus_per_node"],
        strategy=config["strategy"],
        num_nodes=config["num_nodes"],
        callbacks=[writer_callback],
        enable_progress_bar=config["enable_progress_bar"],
    )

    # Enable time measurement
    if torch.cuda.is_available():
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        starter.record()

    def get_best_model(model_path):
        for file in os.listdir(model_path):
            if file.startswith("ckpt"):
                return file
        # return None

    model_path = Path(
        config["outputs"]["out_folder"],
        config["outputs"]["out_model_name"],
        "checkpoints",
    )

    ckpt_path = os.path.join(model_path, get_best_model(model_path))
    trainer.predict(
        seg_module,
        datamodule=data_module,
        return_predictions=False,
        ckpt_path=ckpt_path,
    )
    truth_msk = config['data']['path_labels_test']
    pred_msk  = os.path.join(out_dir, "predictions"+"_"+config["outputs"]["out_model_name"])

    mIou, ious = generate_miou(config, truth_msk, pred_msk)
    mf1, f1s, oa = generate_mf1s(truth_msk, pred_msk)

    print_iou_metrics(mIou, ious)
    print_f1_metrics(mf1, f1s)
    print_overall_accuracy(oa)

    # if torch.cuda.is_available():
    #     if config["strategy"] != None:
    #         dist.barrier()
    #         torch.cuda.synchronize()
    #     ender.record()
    #     torch.cuda.empty_cache()

    #     inference_time_seconds = starter.elapsed_time(ender) / 1000.0
    #     print_inference_time(inference_time_seconds, config)

    @rank_zero_only
    def print_finish():
        print("--  [FINISHED.]  --", f"output dir : {out_dir}", sep="\n")

    print_finish()

    # # Compute mIoU over the predictions - not done here as the test labels are not available, but if needed, you can use the generate_miou function from metrics.py

    # truth_msk = config['data']['path_labels_test']
    # pred_msk  = os.path.join(out_dir, "predictions"+"_"+config["outputs"]["out_model_name"])

    # mIou, ious = generate_miou(config, truth_msk, pred_msk)
    # mf1, f1s, oa = generate_mf1s(truth_msk, pred_msk)

    # print_iou_metrics(mIou, ious)
    # print_f1_metrics(mf1, f1s)
    # print_overall_accuracy(oa)

if __name__ == "__main__":
    args = argParser.parse_args()

    config = read_config(args.config_file)

    assert config["num_classes"] == config["out_conv"][-1]

    print_config(config)
    main(config)
