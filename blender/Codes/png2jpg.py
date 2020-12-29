from PIL import Image
import os

input_path = 'C:/Users/Lab/GoogleDrive/Database/ds_002/train/'
output_path = 'C:/Users/Lab/Documents/a/'
files = os.listdir(input_path)
count = 1

for file in files:
    if file[-4:] == ".png":
        input_im = Image.open(input_path + "img_{:05d}.png".format(count))
        rgb_im = input_im.convert('RGB')
        rgb_im.save(output_path + "img_{:05d}.jpg".format(count), quality=100)
        count = count + 1

print("Process finished!")