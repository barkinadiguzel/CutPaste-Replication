# in training mode 
import torch

def forward_pass(image, backbone, embedding_head=None, classifier=None, cutpaste_aug=None):
    if cutpaste_aug is not None:
        image_aug = cutpaste_aug(image)
    else:
        image_aug = image

    x = backbone(image_aug.unsqueeze(0))  
    x = x.flatten(1)

    if embedding_head is not None:
        embedding = embedding_head(x)
    else:
        embedding = x

    if classifier is not None:
        logits = classifier(x)
    else:
        logits = None

    return embedding, logits
