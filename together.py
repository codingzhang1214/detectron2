
#如果你在两次不同的Python进程中分别执行了数据集注册脚本和训练脚本，那么即使是先执行了数据集注册，再执行训练脚本，数据集注册信息也不会保留到训练脚本的执行过程中。
#这是因为Detectron2（和许多其他Python库）在内存中管理注册信息，这些信息只在当前Python进程的生命周期内有效。当Python进程结束时，所有在该进程中注册的数据集信息都会丢失，不会被传递到下一个进程。
#因此，为了确保训练脚本能够识别到你注册的数据集，你需要在执行训练代码之前，在同一个Python进程（即在同一个脚本或Python会话中）执行数据集注册代码。

from detectron2.data.datasets import register_pascal_voc
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import os

def register_fish_dataset():
    register_pascal_voc(
        name="fish_train",
        dirname="/mnt/DMlab/zdx/data_set/Fish",
        split="train",
        year=2007,
        class_names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
    )
    register_pascal_voc(
        name="fish_val",
        dirname="/mnt/DMlab/zdx/data_set/Fish",
        split="val",
        year=2007,
        class_names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
    )
    register_pascal_voc(
        name="fish_test",
        dirname="/mnt/DMlab/zdx/data_set/Fish",
        split="test",
        year=2007,
        class_names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
    )
    register_pascal_voc(
        name="fish_trainval",
        dirname="/mnt/DMlab/zdx/data_set/Fish",
        split="trainval",
        year=2007,
        class_names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
    )

if __name__ == "__main__":
    # 注册数据集
    register_fish_dataset()

    # 配置日志
    setup_logger()

    # 配置Detectron2
    cfg = get_cfg()
    cfg.merge_from_file("/home/zdx/PythonProjects/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("fish_train",)
    cfg.DATASETS.TEST = ("fish_val",)  # 如果你想在训练时进行验证，请确保这里不是空的
    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.MAX_ITER = 100000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 14  # 更新为你的类别数

    # 指定输出目录
    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 训练
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
