import argparse
import os
import time

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract

import easyocr

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
        image = np.array(Image.open(filename))
        
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


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--images_path', required=True)
    ap.add_argument('--gpu', default='-1')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    te = TextExtractor()

    for f in os.listdir(args.images_path):
        filename = os.path.join(args.images_path, f)
        print()
        print(filename)

        start = time.time()
        text = te.precise_ocr(filename)
        print(text)
        print(f"Inference took: {time.time()-start} s.")
        print()
