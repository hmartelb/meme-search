import argparse
import os
import time

import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import torch
import torchvision
from PIL import Image


class TextExtractor():
    def __init__(self, 
        languages=['en', 'ch_sim'],
        pytesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    ):
        self.languages = languages
        self.reader = easyocr.Reader(languages)
        pytesseract.pytesseract.tesseract_cmd = pytesseract_path
        
    def fast_ocr(self, filename):
        '''
        [CPU only]
        Uses Pytesseract and OpenCV to extract text from the image. 
        Faster but not as accurate as the precise_ocr() method. 
        '''
        image = np.array(Image.open(filename).convert('RGB'))
        
        # OpenCV preprocessing inspired from:
        # https://towardsdatascience.com/extract-text-from-memes-with-python-opencv-tesseract-ocr-63c2ccd72b69
        image = cv2.bilateralFilter(image, 5, 55, 60)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 240, 255, 1) 
        
        text = pytesseract.image_to_string(image)
        text = list(map(str.lower, text.split()))
        return text

    def precise_ocr(self, filename):
        '''
        [GPU support]
        Uses EasyOCR (PyTorch-based) to extract text from the image.
        Slower than the fast_ocr() method but more accurate. 
        '''
        text = self.reader.readtext(filename, detail=0)
        text = (' '.join(text)).split()
        return text


class ImageExtractor():
    def __init__(self):
        # Load the pretrained model
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.eval()
        self.layer = self.model._modules.get('avgpool')
        
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def to_vec(self, filename):
        '''
        https://stackoverflow.com/questions/63552044/how-to-extract-feature-vector-from-single-image-in-pytorch
        '''
        img_embedding = torch.zeros(512)
        image = self.transforms(Image.open(filename).convert('RGB'))

        # Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            img_embedding.copy_(o.flatten())
        # Attach that function to our selected layer
        h = self.layer.register_forward_hook(copy_data)

        with torch.no_grad():                               
            self.model(image.unsqueeze(0))

        # Detach our copy function from the layer
        h.remove()
        return img_embedding


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--images_path', required=True)
    ap.add_argument('--gpu', default='-1')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    te = TextExtractor()
    ie = ImageExtractor()

    for f in os.listdir(args.images_path):
        filename = os.path.join(args.images_path, f)
        print()
        print(filename)

        start = time.time()
        # text = te.precise_ocr(filename)
        # print(text)
        img_embedding = ie.to_vec(filename)
        print(img_embedding.cpu().numpy().shape)
        print(f"Inference took: {time.time()-start} s.")
        print()
