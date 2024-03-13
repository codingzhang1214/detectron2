from detectron2.data.datasets import register_pascal_voc

def register_fish_dataset():
    register_pascal_voc(
        name="fish_train",
        dirname="/mnt/DMlab/zdx/data set/Fish",
        split="train",  # 这里指定使用 train.txt 文件
        year=2007,  # VOC 数据集的年份，只是示例，你可以根据需要更改
        class_names=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14' ],  # 替换为你的实际类别
    )

    register_pascal_voc(
        name="fish_val",
        dirname="/mnt/DMlab/zdx/data set/Fish",
        split="val",  # 这里指定使用 val.txt 文件
        year=2007,
        class_names=["__background__", "class1", "class2", ...],
    )

    register_pascal_voc(
        name="fish_test",
        dirname="/mnt/DMlab/zdx/data set/Fish",
        split="test",  # 这里指定使用 test.txt 文件
        year=2007,
        class_names=["__background__", "class1", "class2", ...],
    )

    register_pascal_voc(
        name="fish_trainval",
        dirname="/mnt/DMlab/zdx/data set/Fish",
        split="trainval",  # 这里指定使用 trainval.txt 文件
        year=2007,
        class_names=["__background__", "class1", "class2", ...],
    )

if __name__ == "__main__":
    register_fish_dataset()
