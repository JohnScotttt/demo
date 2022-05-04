import torch
import cv2
from torch.autograd import Variable
from torchvision import transforms
from models.cnn import Net

use_cuda = True
model = Net()
model.load_state_dict(torch.load('output/params_30.pth'))
# modelt = torch.load('output/params_30.pth')
model.eval()
if use_cuda and torch.cuda.is_available():
    model.cuda()

# to_onnx(model, 3, 28, 28, 'output/params.onnx')

img = cv2.imread('data/test_images/7_00514.jpg')
img_tensor = transforms.ToTensor()(img)
img_tensor = img_tensor.unsqueeze(0)
if use_cuda and torch.cuda.is_available():
    prediction = model(Variable(img_tensor.cuda()))
else:
    prediction = model(Variable(img_tensor))
pred =torch.max(prediction,1)[1]
print(pred.item())
cv2.imshow("image", img)
cv2.waitKey(0)