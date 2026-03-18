import random

def random_flip(image):
    if random.random() > 0.5:
        return image.transpose(method=Image.FLIP_LEFT_RIGHT)
    return image
