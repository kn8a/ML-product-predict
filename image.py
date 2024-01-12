# image.py
from PIL import Image
import webcolors

def process_image(product, color):
    # Load image
    img = Image.open("./assets/%s.png" % product)

    # Get RGB
    if color == "Peach":
        rgb = webcolors.name_to_rgb("peachpuff")
    elif color == "Charcoal":
        rgb = webcolors.name_to_rgb("dimgray")
    else:
        rgb = webcolors.name_to_rgb(color)

    # Change color
    # Get image size
    width, height = img.size

    # Iterate through each pixel and recolor red pixels
    for x in range(width):
        for y in range(height):
            pixel_color = img.getpixel((x, y))

            # Check if the pixel is red
            if pixel_color[0] == 237 and pixel_color[1] == 28 and pixel_color[2] == 36:
                # Recolor the pixel with the specified color
                img.putpixel((x, y), rgb)

    # Save output
    img.save("./assets/output.png")