import PIL
import numpy as np
import torch
from torchvision.models import resnet50

from grad_cam import GradCAM
from utils import show_cam_on_image

# 所有的代码摘抄自这个库
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from torchvision.models import resnet50

model = resnet50(pretrained=True)
target_layers = [model.layer4[-1]]  # 输出feature map的层，这里是resnet最后一层
'''
[Bottleneck(
   (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
   (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
   (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
   (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
   (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
   (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
   (relu): ReLU(inplace=True)
 )]
 '''
img_path = "./test_img.jpg"
pil_img = PIL.Image.open(img_path)
rgb_img = np.asarray(pil_img) / 255
input_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float()  # 1, 3, 1069, 1070
# input_tensor = F.upsample(input_tensor, size=(224, 224), mode='bilinear', align_corners=False)

# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# If target_category is None, the highest scoring category
# will be used for every image in the batch.
# target_category can also be an integer, or a list of different integers
# for every image in the batch.
# ------------
# target_category = 5
target_category = None
# target_category = [5, 6]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # (1, 1069, 1070)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]  # h, w
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)  # uint8 # (1069, 1070, 3)

print(visualization.shape)  # (1069, 1070, 3)
save_img = PIL.Image.fromarray(visualization)
save_img.save("visualization.jpg")
