import os
import pickle
import string
import time

import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import torch
import torchvision
from PIL import Image
from sentence_transformers import SentenceTransformer


class SentenceVectorizer():
    def __init__(self, filename=None, dim=0, eps=1e-9):
        self.dim = dim
        self.eps = eps
        self.word2vec = {}

        if filename is not None:
            self.load(filename)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.word2vec = pickle.load(f)
            self.dim = len(next(iter(self.word2vec.values())))

    def encode(self, text):
        '''
        Sum the word vectors for each token in the sentence. Does not take order nor syntax into account, but it is very fast.
        Inspired from this answer on stackoverflow:
        https://stackoverflow.com/questions/30795944/how-can-a-sentence-or-a-document-be-converted-to-a-vector
        '''
        assert type(text) == str, "Wrong data type, text must be str"
        text = self._tokenize(text)
        vec = np.zeros(self.dim)
        for word in text:
            try:
                vec = np.add(vec, self.word2vec[word])
            except:
                pass # Out of vocabulary (OOV)
        vec /= (np.sqrt(vec.dot(vec)) + self.eps)
        return vec

    def _tokenize(self, text):
        return text.lower()\
            .strip(string.punctuation)\
            .split(' ')


class TextExtractor():
    def __init__(self, 
        languages=['en'],#, 'ch_sim'],
        pytesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    ):
        self.languages = languages
        self.reader = easyocr.Reader(languages)
        self.sentence_transformer = SentenceTransformer('bert-base-nli-mean-tokens') # stsb-distilbert-base, bert-base-nli-mean-tokens, paraphrase-distilroberta-base-v1
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
        # text = list(map(str.lower, text.split()))
        return text

    def precise_ocr(self, filename):
        '''
        [GPU support]
        Uses EasyOCR (PyTorch-based) to extract text from the image.
        Slower than the fast_ocr() method but more accurate. 
        '''
        text = self.reader.readtext(filename, detail=0)
        # text = list(map(str.lower,' '.join(text).split())
        return text

    def to_vec(self, filename=None, text=None, method='precise', to_numpy=False, return_text=False):
        '''
        Use the sentence transformers package to get sentence embeddings.
        https://github.com/UKPLab/sentence-transformers
        '''
        if filename is not None:
            ocr_fn = self.fast_ocr if method == 'fast' else self.precise_ocr
            text = ocr_fn(filename)

        text = ' '.join(text)
        if type(text) == str:
            text = [text]

        sentence_embedding = self.sentence_transformer.encode(text)
        if sentence_embedding.ndim > 1:
            sentence_embedding = np.mean(sentence_embedding, axis=0)
        
        # By default, the returned value is a numpy array. Convert to tensor
        if not to_numpy:
            sentence_embedding = torch.squeeze(torch.from_numpy(sentence_embedding))
        
        if return_text:
            # return sentence_embedding[0], text
            return sentence_embedding, text
        return sentence_embedding


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

        self.cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def to_vec(self, filename, to_numpy=False):
        '''
        https://stackoverflow.com/questions/63552044/how-to-extract-feature-vector-from-single-image-in-pytorch

        TODO: Consider this approach
        https://github.com/erezposner/Fast_Dense_Feature_Extraction
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
        # img_embedding = img_embedding.unsqueeze(0)
        if to_numpy:
            img_embedding = img_embedding.cpu().numpy()
        return img_embedding

    def cosine_similarity(self, vec1, vec2, to_numpy=False):
        score = self.cos_sim(vec1, vec2)
        if to_numpy:
            return np.asscalar(score.cpu().numpy())
        return score


def similarity_matrix(vectors):
    M = np.zeros([len(vectors), len(vectors)])
    for i in range(len(vectors)):
        for j in range(i, len(vectors)):
            M[i,j] = ie.cosine_similarity(vectors[i], vectors[j], to_numpy=True)
    return M

if __name__ == '__main__':
    import argparse

    import pandas as pd
    from tqdm import tqdm

    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data_dl.tsv')
    ap.add_argument('--output', default='index.pkl')
    ap.add_argument('--downloads_folder', default='images')
    ap.add_argument('--gpu', default='-1')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    sv = SentenceVectorizer(filename=os.path.join('..','api','pretrained','glove.6B.300d_dict.pickle'))
    ie = ImageExtractor()

    df = pd.read_csv(args.input, sep='\t')
    
    df['title_glove'] = None
    df['ocr_glove'] = None
    df['fusion_text_glove'] = None
    df['img_embedding'] = None

    to_drop = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            if type(row['text']) != str:
                row['text'] = ' '

            df['title_glove'].iloc[i] = sv.encode(row['title'])
            df['ocr_glove'].iloc[i] = sv.encode(row['text'])
            df['fusion_text_glove'].iloc[i] = sv.encode(row['title'] + ' ' + row['text'])

            df['img_embedding'].iloc[i] = ie.to_vec(
                filename=os.path.join(args.downloads_folder, row['id']+row['format']),
                to_numpy=True
            )
        except:
            print(f"Error while processing row at index {i}")
            print(row)
            to_drop.append(i)

    if len(to_drop) > 0:
        print(f"Drop {len(to_drop)} entries")
        df.drop(index=to_drop, inplace=True)
    
    print(f"Saving index to {args.output}")
    df.to_pickle(args.output)    

    # Integrity load tests
    df2 = pd.read_pickle(args.output)
    assert df2.iloc[0]['img_embedding'].shape == (512,), f"Error loading img_embedding, shape is {df2.iloc[0]['img_embedding'].shape}"
    assert df2.iloc[0]['fusion_text_glove'].shape == (300,), f"Error loading fusion_text_glove, shape is {df2.iloc[0]['fusion_text_glove'].shape}"

    # df2['url'] = df2['url'].apply(lambda x: 'https://'+ x )
    # df2.to_pickle(args.output)