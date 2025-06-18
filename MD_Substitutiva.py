from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('imagem_original.png').convert('L')
img_array = np.array(img)

# Exemplo de criação de marca d'água aleatória
marca_dim = (64, 64)
marca = np.random.randint(0, 2, size=marca_dim, dtype=np.uint8)

def embed_lsb(image, watermark, top=0, left=0):
    img_copy = image.copy()
    h, w = watermark.shape
    for i in range(h):
        for j in range(w):
            pixel = img_copy[top+i, left+j]
            pixel = (pixel & 0xFE) | watermark[i, j]
            img_copy[top+i, left+j] = pixel

            return img_copy

def extract_lsb(image, marca_shape, top=0, left=0):
    h, w = marca_shape
    extracted = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            extracted[i, j] = image[top+i, left+j] & 1

            return extracted


img_watermarked = embed_lsb(img_array, marca, top=0, left=0)

img_emb = Image.fromarray(img_watermarked)

img_emb.save('imagem_marcada.png')

marca_extraida = extract_lsb(img_watermarked, marca.shape, top=0,
left=0)
plt.imshow(marca_extraida, cmap='gray')
plt.title("Marca d'água extraída")
plt.show()