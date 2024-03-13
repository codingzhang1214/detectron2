import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

# 为了能看到训练过程中的日志输出
setup_logger()

# 加载配置文件
cfg = get_cfg()
cfg.merge_from_file("/home/zdx/PythonProjects/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

# 如果你有多个GPU，这里可以帮助你更有效地利用它们
#cfg.DATASETS.TRAIN = ("fish_train",)
#cfg.DATASETS.TEST = ("fish_val",)
#cfg.DATALOADER.NUM_WORKERS = 4
#cfg.SOLVER.IMS_PER_BATCH = 2  # 根据你的GPU调整批次大小
#cfg.SOLVER.BASE_LR = 0.0025  # 可能需要根据你的数据集和模型调整学习率
#cfg.SOLVER.MAX_ITER = 10000  # 根据你的需要调整迭代次数
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # 包括背景的类别总数

# 指定输出目录
cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
