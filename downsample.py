from PIL import Image

img = Image.open("/home/james/Documents/fireHoseSam/mapfiles/output/0_child_match.png").convert("L")  # "L" for grayscale

downsampled = img.resize((128, 128), resample=Image.BOX)
downsampled.save("downsampled.png")
downsampled.show()