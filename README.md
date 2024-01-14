## 1、环境

```python
# 激活环境

conda activate detector

cd path/to/CornerNet
```



## 2、训练

```python
python train.py
```

可以调整的参数如下：

```python
Cuda = True		# 是否使用CUDA
fp16 = False	# 是否使用混合精度训练, 可减少约一半的显存，但会一定程度上影响精度
classes_path = 'config/classes_path.txt'	# 记录类别的文件，角点检测中只有‘point’一类，不用关注
model_path = 'logs/saved/mobilenet_v3_v4_SA/best_val_weights.pth'  # 自定义预训练权重
input_shape = [640, 640]	# 模型输入大小，256,512,640,...等均可

backbone = "mobilenetv3"	# 模型采用的主干，可选resnet50、mobilenetv3
pretrained = False	#是否加载官方预训练权重（仅支持resnet）

# train on two stages
Init_Epoch = 0	# 初始epoch
Freeze_Epoch = 100	# 第一阶段训练的epoch数
Freeze_batch_size = 16 # 第一阶段训练采用的batch size

UnFreeze_Epoch = 200	# 第二阶段训练的epoch数
Unfreeze_batch_size = 16	# 第二阶段训练采用的batch size

Freeze_Train = True		# 是否采用两阶段训练

Init_lr = 1e-4		# 初始学习率
Min_lr = Init_lr * 0.01		# 最小学习率，默认为初始的0.01，放置动态学习率调整后的学习率过小

optimizer_type = "adam"		# 优化器，可选sgd、adam
momentum = 0.9		# 动量
weight_decay = 5e-4		# 权值衰减

lr_decay_type = 'cos'	# 动态学习率，默认采用余弦退火

save_period = 1e3	# 每隔多少epoch保存一次权重，默认不保存（1e3 > UnFreeze_Epoch）
save_dir = 'logs'	# 存储的log路径，log路径下会保存loss、训练过程的log、最优训练权重和最优验证权重等

num_workers = 4		# 线程数

train_annotation_path = 'config/total_v3_v4.txt'	# 训练数据集
val_annotation_path = 'config/total_v3.txt'		# 验证数据集
```

## 3、评估

```python
python eval.py
```

可以调整的参数如下：

```python
#   map_mode为0代表整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map
#   map_mode为1代表仅仅获得预测结果
#   map_mode为2代表仅仅获得真实框
#   map_mode为3代表仅仅计算VOC_map
#   map_mode为4代表计算当前数据集COCO的0.50:0.95map
map_mode = 0

MINOVERLAP = 0.25	# 阈值，0.25表示评估mAP@.25, 由于角点检测没有bbox，这里默认设置了一个6px的bbox，评估mAP@.25约相当于精度在4px以内的为正样本

map_out_path = 'logs/map_out/mobilenet_v3_v4_SA@.25'	# 结果输出路径
eval_f = 'config/total_v4.txt'		# 评估的数据集文件
```

## 4、测试

```python
python predict.py
```
可以调整的参数如下：
```python
# predict.py

#   mode用于指定测试的模式：
#   'predict'           表示单张图片预测
#   'video'             表示视频检测，可调用摄像头或者视频进行检测
#   'fps'               表示测试fps，使用的图片是img里面的street.jpg
#   'dir_predict'       表示遍历文件夹进行检测并保存
#   'heatmap'           表示进行预测结果的热力图可视化
#   'export_onnx'       表示将模型导出为onnx
mode = "predict"

crop = False	#  是否在单张图片预测后对目标进行截取
count = False	#  是否进行目标的计数

#   video_path          指定视频的路径，当video_path=0时表示检测摄像头
#                       检测视频，则设置video_path = "xxx.mp4"
#   video_save_path     视频保存的路径，当video_save_path=""时表示不保存
#   video_fps           用于保存的视频的fps
video_path = 'test/output_5fps.mp4'
video_save_path = "test/output_5fps_detect_mobile.mp4"
video_fps = 30.0

#   test_interval       用于指定测量fps的时候，图片检测的次数，默认100
#   fps_image_path      用于指定测试的fps图片
test_interval = 100
fps_image_path = "test/test1_copy.jpg"

dir_origin_path = "dataset/20221215_hyh_rgb_part0_v1/img/"	# 用于检测的图片的文件夹路径
dir_save_path = "dataset/20221215_hyh_rgb_part0_v1/img_pred/"	# 检测完图片文件夹的保存路径

heatmap_save_path = "test/test2_heatmap_640.jpg"	# 热力图的保存路径，仅在mode='heatmap'有效

simplify = True
onnx_save_path = "nets/cornernet_mobilenetv3_SA.onnx"	# 模型导出路径
```

```python
# detect_config.py

_defaults = {
        "classes_path": 'config/classes_path.txt',
		
    	# 测试使用的网络权重文件、backbone和置信度阈值
        "model_path": 'logs/saved/mobilenet_v3_v4_SA/best_val_weights.pth',
        "backbone": 'mobilenetv3',
        "confidence": 0.42,
		
    	# 模型输入、是否采用nms（默认不采用）和采用nms时的阈值
        "input_shape": [640, 640],
    	"nms": False,
        "nms_iou": 0.9,     
    
    	# 是否对图像进行不失真resize（默认不采用）
        "letterbox_image": False,
    
    	# 使用GPU还是CPU预测
        "cuda": True
    }
```

