from PIL import Image, ImageSequence
import numpy as np
import imageio
import os

def load_frames(image: Image, mode='RGB'):
    return np.array([
        np.array(frame.convert(mode))
        for frame in ImageSequence.Iterator(image)
    ])

gifs = []
count = 0
for filename in os.listdir("/Users/emilygu/Desktop/CRAFT/sample_gifs/"):
    print(filename)
    if (count == 10): break
    count += 1
    f = os.path.join("/Users/emilygu/Desktop/CRAFT/sample_gifs/", filename)
    with Image.open(f) as im:
        frames = load_frames(im)
    if (frames.shape[0] == 8):
        gifs.append(frames)

gifs = np.array(gifs)
display = []
for k in range(8):
    display.append(np.concatenate(gifs[:, k, :, :, :], axis=1))

display = np.array(display)
print(display.shape)

imageio.mimsave("display.gif", display, "GIF")