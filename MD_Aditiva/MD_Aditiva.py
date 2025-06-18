from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('imagem_original.png').convert('L')
img_array = np.array(img)

# Marca binária: 0 ou 1
marca_binaria = np.random.randint(0, 2, size=(64, 64), dtype=np.uint8)
# Converte para {-1, +1}
marca = 2 * marca_binaria - 1

def embed_additive(image, watermark, alpha, top=0, left=0):
    img_copy = image.astype(np.float32).copy()
    h, w = watermark.shape
    for i in range(h):
        for j in range(w):
            pixel = img_copy[top + i, left + j]
            pixel_marked = pixel + alpha * watermark[i, j]
            # Garante que o pixel esteja dentro do intervalo [0, 255]
            img_copy[top + i, left + j] = np.clip(pixel_marked, 0, 255)

    return img_copy.astype(np.uint8)

img_watermarked = embed_additive(img_array, marca, alpha=10, top=0,left=0)
Image.fromarray(img_watermarked).save('imagem_marcada.png')


def extract_additive(original, marked, marca_shape, alpha, top=0,left=0):
    h, w = marca_shape
    extracted = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            diff = marked[top+i, left+j] - original[top+i, left+j]
            w_hat = 1 if diff >= 0 else -1
            extracted[i, j] = 1 if w_hat == 1 else 0

    return extracted

marca_extraida = extract_additive(img_array, img_watermarked,
marca.shape, alpha=10)
plt.imshow(marca_extraida, cmap='gray')
plt.title("Marca d’água extraída")
plt.show()