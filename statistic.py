import cv2
import numpy as np
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore


# 用来为多线程做数据汇总，记录一个数据集中的信息
classes_pixelnum = dict()
pixelnum = 0.0
# 用来控制多线程的信号量
data_lock = None        # <-- 防止控制上述变量互斥
func_lock = None        # <-- 记录剩余未完成线程个数
funcfinal_lock = None   # <-- 保证多线程记录同一个数据集
tqdm_lock = None        # <-- 控制进度条

labels_info = [
    {"name": "Sky", "id": 0, "color": [128, 128, 128], "trainId": 0},
    {"name": "Bridge", "id": 1, "color": [0, 128, 64], "trainId": 1},
    {"name": "Building", "id": 2, "color": [128, 0, 0], "trainId": 1},
    {"name": "Wall", "id": 3, "color": [64, 192, 0], "trainId": 11},
    {"name": "Tunnel", "id": 4, "color": [64, 0, 64], "trainId": 1},
    {"name": "Archway", "id": 5, "color": [192, 0, 128], "trainId": 1},
    {"name": "Column_Pole", "id": 6, "color": [192, 192, 128], "trainId": 2},
    {"name": "TrafficCone", "id": 7, "color": [0, 0, 64], "trainId": 2},
    {"name": "Road", "id": 8, "color": [128, 64, 128], "trainId": 3},
    {"name": "LaneMkgsDriv", "id": 9, "color": [128, 0, 192], "trainId": 3},
    {"name": "LaneMkgsNonDriv", "id": 10, "color": [192, 0, 64], "trainId": 3},
    {"name": "Sidewalk", "id": 11, "color": [0, 0, 192], "trainId": 4},
    {"name": "ParkingBlock", "id": 12, "color": [64, 192, 128], "trainId": 4},
    {"name": "RoadShoulder", "id": 13, "color": [128, 128, 192], "trainId": 4},
    {"name": "Tree", "id": 14, "color": [128, 128, 0], "trainId": 5},
    {"name": "VegetationMisc", "id": 15, "color": [192, 192, 0], "trainId": 5},
    {"name": "SignSymbol", "id": 16, "color": [192, 128, 128], "trainId": 6},
    {"name": "Misc_Text", "id": 17, "color": [128, 128, 64], "trainId": 6},
    {"name": "TrafficLight", "id": 18, "color": [0, 64, 64], "trainId": 6},
    {"name": "Fence", "id": 19, "color": [64, 64, 128], "trainId": 7},
    {"name": "Car", "id": 20, "color": [64, 0, 128], "trainId": 8},
    {"name": "SUVPickupTruck", "id": 21, "color": [64, 128, 192], "trainId": 8},
    {"name": "Truck_Bus", "id": 22, "color": [192, 128, 192], "trainId": 8},
    {"name": "Train", "id": 23, "color": [192, 64, 128], "trainId": 8},
    {"name": "OtherMoving", "id": 24, "color": [128, 64, 64], "trainId": 8},
    {"name": "Pedestrian", "id": 25, "color": [64, 64, 0], "trainId":9},
    {"name": "Child", "id": 26, "color": [192, 128, 64], "trainId":9},
    {"name": "CartLuggagePram", "id": 27, "color": [64, 0, 192], "trainId": 9},
    {"name": "Animal", "id": 28, "color": [64, 128, 64], "trainId": 9},
    {"name": "Bicyclist", "id": 29, "color": [0, 128, 192], "trainId": 10},
    {"name": "MotorcycleScooter", "id": 30, "color": [192, 0, 192], "trainId": 10},
    {"name": "Void", "id": 31, "color": [0, 0, 0], "trainId": 255}
]


colors = []
for el in labels_info:
    (r, g, b) = el['color']
    colors.append((r, g, b))
    
color2id = dict(zip(colors, range(len(colors))))

def convert_labels(color2id, label):
    mask = np.full(label.shape[:2], 2, dtype=np.uint8)
    # mask = np.zeros(label.shape[:2])
    for k, v in color2id.items():
        mask[cv2.inRange(label, np.array(k) - 1, np.array(k) + 1) == 255] = v
        
    return mask

def thread_pool_callback(worker):
    """
    `thread_pool_callback` is a callback function that is called when a worker thread in the thread pool
    executor completes. It is used to traceback worker exceptions, which threadpool don't trace.
    
    Args:
      worker: The worker object that is running the task.
    """
    # logger.info("called thread pool executor callback function")
    worker_exception: AttributeError = worker.exception()
    if worker_exception:
        print(worker_exception.with_traceback())
        # logger.exception("Worker return exception: {}".format(worker_exception))

def statistic_onelabel(label_path: str, use_threadpool: bool = True):
    """
    > The function is used to calculate the number of pixels of each class in the label image
    
    Args:
      label_path (str): the path of the label image
      use_threadpool (bool): Whether to use threadpool to speed up the calculation. Defaults to True
    """
    # label = cv2.imread(label_path)[..., 0]
    label = cv2.imread(label_path)[:, :, ::-1]
    label = convert_labels(color2id, label)
    global classes_pixelnum 
    global pixelnum
    global data_lock
    global func_lock
    global funcfinal_lock
    global tqdm_lock
    for statistic_classid in statistic_classes:
        # Here you get the mask corresponding to a certain class in the label.
        mask = label == statistic_classid
        if use_threadpool:
            data_lock.acquire()
        # NOTE: The following two lines are mainly used in the entire code.
        classes_pixelnum[statistic_classid] += float(np.sum(mask))     # <-- Here the number of pixels corresponding to a certain class in an image will be calculated.
        pixelnum += float(np.ones_like(mask).sum())                    # <-- Here it is recorded how many pixels are in an image.
        if use_threadpool:
            data_lock.release()
    if use_threadpool:
        func_lock.acquire()
        if func_lock._value == 0:
            funcfinal_lock.release()
        tqdm_lock.release()
    # print(classes_pixelnum)

def statistic_split(split_path: str, statistic_classes: list, use_threadpool: bool = True):
    """
    It reads the split file, and then for each label path, it calls the function statistic_onelabel
    
    Args:
      split_path (str): the path to the split file
      statistic_classes (list): list of class ids to be counted
      use_threadpool (bool): whether to use threadpool to speed up the process. Defaults to True
    
    Returns:
      classes_pixelnum, pixelnum, len(label_paths)
    """
    label_paths = list()
    with open(split_path) as sf:
        image_and_label_pathstrs = sf.readlines()
        for i_and_l_pstr in image_and_label_pathstrs:
            image_path, label_path = i_and_l_pstr.strip().split(" ")

            
            if "CamVid" in split_path:
                label_path = 'D:/Study/code/CamVid/' + label_path
            else:
                label_path = 'D:/Study/code/' + label_path
                
            if "camera_lidar_semantic" in label_path:
                label_path = label_path.replace("label", "mask")
            label_paths.append(label_path)

    global classes_pixelnum
    global pixelnum
    classes_pixelnum = {c: 0 for c in statistic_classes}
    pixelnum = 0

    if use_threadpool:
        tpool = ThreadPoolExecutor(128)
        global data_lock
        global func_lock
        global funcfinal_lock
        global tqdm_lock
        data_lock = Semaphore(1)
        func_lock = Semaphore(len(label_paths))
        funcfinal_lock = Semaphore(0)
        tqdm_lock = Semaphore(256)
    for label_path in tqdm(label_paths):
        if use_threadpool:
            tqdm_lock.acquire()
            thread_pool_exc = tpool.submit(statistic_onelabel, label_path)
            thread_pool_exc.add_done_callback(thread_pool_callback)
        else:
            statistic_onelabel(label_path, use_threadpool=False)
            # label = cv2.imread(label_path)[..., 0]
            # for statistic_classid in statistic_classes:
            #     mask = label == statistic_classid
            #     # data_lock.acquire()
            #     classes_pixelnum[statistic_classid] += np.sum(mask)
            #     pixelnum.append(pixelnum.pop(0) + np.ones_like(mask).sum())

    if use_threadpool:
        funcfinal_lock.acquire()
    return classes_pixelnum, pixelnum, len(label_paths)


def print_statistic_result(classes_pixelnum: dict, pixelnum: int, datanum: int):
    """
    It prints the average pixel number of each class in one picture and the percentage of the total pixel number of
    each class in one picture.
    
    Args:
      classes_pixelnum (dict): a dictionary that stores the number of pixels of each class in the
    dataset.
      pixelnum (int): the total number of pixels in all images
      datanum (int): the number of images in the dataset
    """
    print("data magnitude:", datanum, classes_pixelnum)
    for cid, pnum in classes_pixelnum.items():
        print(f"class_id: {cid}, average_pixel_num: {pnum / datanum}, percentage: {(pnum / pixelnum) * 100} %")


if __name__ == "__main__":
    # 此处设置需要统计类别像素数的数据集 lst 文件，每一个 lst 文件中的文件结构如下所示：
    # example: *.lst
    # |--------------------------
    # |...
    # |....../image.png ....../label.png
    # |...
    # |--------------------------
    # 每一行内容为:
    # <图片地址><空格><标签地址><\n>
    # 
    # 其中可包含多个数据集，形式为两层嵌套列表，每一层列表就是一组数据集
    statistic_datasets = [
        # [
        #     "split/fdjwv1_base_mix_train.lst",
        #     "split/fdjwv1_base_mix_val.lst",
        # ],
        # [
        #     "datasets/Cityscapes/train.lst", 
        # ],
        [
            "datasets/CamVid/train.lst",  
        ],
        # [
        #     "split/selecteda2d2_train.lst",
        #     "split/selecteda2d2_val.lst",
        # ],
        # [
        #     "split/train.lst",
        #     "split/val.lst",
        #     "split/test.lst",
        # ],
        # [
        #     "split/cityscapes_train.lst",
        #     "split/cityscapes_val.lst",
        # ],
    ]

    # 此处设置需要统计的类别在标签上对应的 id
    # statistic_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    statistic_classes = range(0,31)

    total_classes_pixelnum = {c: 0 for c in statistic_classes}
    total_pixelnum = 0
    total_datanum = 0
    for s_dataset in statistic_datasets:
        print("-------------------------------------")
        print("run statictic for split:", s_dataset)

        classes_pixelnum = {c: 0 for c in statistic_classes}
        pixelnum = 0
        datanum = 0
        for s_split in s_dataset:
            # 此处统计一个 lst 文件包含的三种信息：
            # s_classes_pixelnum: 每个类别的像素数量
            # s_pixelnum: 所有图片的像素数量（用来计算某类别在整个数据集上的像素占比）
            # s_datanum: 包含的图片数量
            # use_threadpool: 可以设置是否用多线程
            s_classes_pixelnum, s_pixelnum, s_datanum = statistic_split(s_split, statistic_classes, use_threadpool=False)

            # 此处将上述统计的一个 lst 中包含的信息做一下汇总
            for cid, pnum in s_classes_pixelnum.items():
                classes_pixelnum[cid] += pnum
            pixelnum += s_pixelnum
            datanum += s_datanum

        # 输出一个数据集中的类别像素占比信息
        print_statistic_result(classes_pixelnum, pixelnum, datanum)

        # 此处将上述统计的一个数据集中包含的信息做一下汇总
        for cid, pnum in classes_pixelnum.items():
            total_classes_pixelnum[cid] += pnum
        total_pixelnum += pixelnum
        total_datanum += datanum

    # 输出统计的所有数据集中的类别像素占比信息
    print("=============================================")
    print("total result:")
    print_statistic_result(total_classes_pixelnum, total_pixelnum, total_datanum)


