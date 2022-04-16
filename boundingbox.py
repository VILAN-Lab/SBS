import torch

pth = r'data/rcnn_feature/1515.pth'
img = torch.load(pth)

# for key in img.keys():
#     print(key, img[key].shape)
att= img['spatial_feature'][10]
print(att)

x = att[2] - att[0]
# x /= 2
y = att[3] - att[1]
# y /= 2
print(x)
print(y)
