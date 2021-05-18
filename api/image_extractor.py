import imagehash
import numpy as np
import torch
import torchvision
from PIL import Image


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

    def to_hash(self, filename, hash_size=8):
        '''
        Calculate the LSH for near-duplicate image detection, as per the following code:
        https://github.com/mendesk/image-ndd-lsh

        Args:
            filename: the image (path as string) to calculate the signature for
            hash_size: hash size to use, signatures will be of length hash_size^2
    
        Returns:
            signature: Image signature as Numpy array
        '''
        image = Image.open(filename).convert("L").resize((hash_size+1, hash_size), Image.ANTIALIAS)
        dhash = imagehash.dhash(image, hash_size)
        signature = dhash.hash.flatten()
        return signature

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
