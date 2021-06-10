from collections import OrderedDict
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

from train_new import resnet50_cbam

def init_cls_model(checkpoint_path, is_multi_gpu=False, classes=2):

    my_model = resnet50_cbam(num_classes=classes)
    state_dict = torch.load(checkpoint_path)['state_dict']
    if is_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        my_model.load_state_dict(new_state_dict)
    else:
        my_model.load_state_dict(state_dict)

    my_model = my_model.cuda()
    my_model.eval()

    return my_model

class Concat_patch(object):  # 切图，实际可以不用
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, margin_ratio=(0.25, 0.25)):
        self.margin_ratio = margin_ratio

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        img = img
        array_img = np.array(img)
        h, w, c = array_img.shape
        h_margin = int(h * self.margin_ratio[0])
        w_margin = int(w * self.margin_ratio[1])
        patches = [array_img[0:h_margin, 0:w_margin, :], array_img[h - h_margin:, 0:w_margin, :],
                   array_img[0:h_margin, w - w_margin:, :], array_img[h - h_margin:, w - w_margin:, :]]

        def concat_patches(patches):
            a = np.concatenate(patches[:2], axis=0)
            b = np.concatenate(patches[2:], axis=0)
            c = np.concatenate([a, b], axis=1)
            return c

        img = concat_patches(patches)
        img = Image.fromarray(img)
        return img

    def __repr__(self):
        interpolate_str = 'reconcat'
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

def cls_judge(img_path, model, img_size=224):
    FALSE_NAME = 'FALSE'
    NG_NAME = 'NG'

    CLS_NAME = [FALSE_NAME, NG_NAME]
    data_transform = transforms.Compose([
        Concat_patch(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])



    file_path = img_path

    with torch.no_grad():
        img_tensor = data_transform(Image.open(file_path).convert('RGB')).unsqueeze(0)
        img_tensor = Variable(img_tensor.cuda(), volatile=True)
        output = F.softmax(model(img_tensor), dim=1).cpu().numpy()
    # defect_prob = round(output.data[0, 1], 6)
    pred = np.argmax(output)
    pred = CLS_NAME[pred]

    score = np.max(output)
    if pred == FALSE_NAME:
        score = 0
    if score <= 0.85 and pred == NG_NAME:
        pred = FALSE_NAME
        score = 0

    return pred, score


if __name__ == '__main__':
    model_path=r'E:\code_tj\CBAM_PyTorch\datasets\work_dir\checkpoint\resnet50-cbam\Models_epoch_0.ckpt'
    img=r'E:\code_tj\CBAM_PyTorch\datasets\val\v06\W0C2P0206A0108_WHITE_20210125.jpg'
    model=init_cls_model(model_path, is_multi_gpu=False, classes=2)
    pre=cls_judge(img, model, img_size=224)
    print(pre)
