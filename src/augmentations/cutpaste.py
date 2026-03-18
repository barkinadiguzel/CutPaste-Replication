import random
from PIL import Image

def cutpaste(image, patch_ratio=0.2, patch_type="normal"):
    w, h = image.size
    if patch_type == "scar":
        pw, ph = int(w*0.05), int(h*0.2)
    else:
        side = int((w*h*patch_ratio)**0.5)
        pw, ph = side, side
      
    x = random.randint(0, w - pw)
    y = random.randint(0, h - ph)
    patch = image.crop((x, y, x+pw, y+ph))
    patch = patch.rotate(random.randint(0, 360))
  
    nx = random.randint(0, w - pw)
    ny = random.randint(0, h - ph)
    image.paste(patch, (nx, ny))
    return image
