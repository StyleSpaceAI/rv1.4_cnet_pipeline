from PIL import Image, ImageOps

def prepare_image(image, size=(512, 512)):
    image = image.convert("RGB")
    image = ImageOps.exif_transpose(image)
    image.thumbnail(size, Image.LANCZOS)

    if image.width % 8 != 0 or image.height % 8 != 0:
        image = crop(image)

    return image


def crop(image):
    height = (image.height // 8) * 8
    width = (image.width // 8) * 8
    left = int((image.width - width) / 2)
    right = left + width
    top = int((image.height - height) / 2)
    bottom = top + height
    image = image.crop((left, top, right, bottom))
    return image
