from PIL import Image

img = Image.new("RGB", (100,255), "black") # create a new black image
pixels = img.load() # create the pixel map

for i in range(img.size[0]):    # for every pixel:
    for j in range(img.size[1]):
        pixels[i,j] = (255, 255, 255) # set the colour accordingly
        print pixels[i,j]

img.save("data/test1.bmp")