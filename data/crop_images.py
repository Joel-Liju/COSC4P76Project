# Improting Image class from PIL module
from PIL import Image
import os


directory = './figures/cropped/'

for filename in os.listdir(directory):
    im = Image.open(directory + filename)
    left = 38
    top = 58
    right = 480
    bottom = 468

    im1 = im.crop((left, top, right, bottom))
    # im1.show()
    im1.save(directory + filename)