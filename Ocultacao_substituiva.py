from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from math import log10, sqrt
import os # Para verificar se o arquivo existe e limpá-lo

# --- Funções LSB (Escala de Cinza) ---
def embed_lsb(image, watermark, top=0, left=0):
    img_copy = image.copy()
    h, w = watermark.shape
    for i in range(h):
        for j in range(w):
            if top + i < img_copy.shape[0] and left + j < img_copy.shape[1]:
                pixel = img_copy[top+i, left+j]
                pixel = (pixel & 0xFE) | watermark[i, j]
                img_copy[top+i, left+j] = pixel
            # else: # Comentado para evitar spam de avisos em execuções normais
            #     print(f"Aviso: Marca d'água excede os limites da imagem em ({top+i}, {left+j}).")
    return img_copy

def extract_lsb(image, marca_shape, top=0, left=0):
    h, w = marca_shape
    extracted = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if top + i < image.shape[0] and left + j < image.shape[1]:
                extracted[i, j] = image[top+i, left+j] & 1
            # else: # Comentado para evitar spam de avisos em execuções normais
            #     print(f"Aviso: Tentativa de extrair fora dos limites da imagem em ({top+i}, {left+j}).")
    return extracted

# --- Funções LSB (RGB) ---
def embed_lsb_rgb(image_rgb, watermark, top=0, left=0, channel=2): # channel 0=R, 1=G, 2=B
    img_copy = image_rgb.copy()
    h, w = watermark.shape
    for i in range(h):
        for j in range(w):
            if top + i < img_copy.shape[0] and left + j < img_copy.shape[1]:
                pixel_channel = img_copy[top+i, left+j, channel]
                pixel_channel = (pixel_channel & 0xFE) | watermark[i, j]
                img_copy[top+i, left+j, channel] = pixel_channel
            # else: # Comentado para evitar spam de avisos em execuções normais
            #     print(f"Aviso RGB: Marca d'água excede os limites da imagem em ({top+i}, {left+j}).")
    return img_copy

def extract_lsb_rgb(image_rgb, marca_shape, top=0, left=0, channel=2): # channel 0=R, 1=G, 2=B
    h, w = marca_shape
    extracted = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if top + i < image_rgb.shape[0] and left + j < image_rgb.shape[1]:
                extracted[i, j] = image_rgb[top+i, left+j, channel] & 1
            # else: # Comentado para evitar spam de avisos em execuções normais
            #     print(f"Aviso RGB: Tentativa de extrair fora dos limites da imagem em ({top+i}, {left+j}).")
    return extracted

# --- Funções de Conversão de Texto para Bits ---
def text_to_bits(text):
    bits = []
    for char in text:
        binary_char = bin(ord(char))[2:].zfill(8)
        for bit in binary_char:
            bits.append(int(bit))
    return np.array(bits, dtype=np.uint8)

def bits_to_text(bits):
    text = ""
    bytes_array = np.array_split(bits, len(bits) // 8)
    for byte_bits in bytes_array:
        if len(byte_bits) == 8:
            byte_int = int("".join(str(b) for b in byte_bits), 2)
            text += chr(byte_int)
    return text

# --- Função de Cálculo de PSNR ---
def calculate_psnr(original, marked):
    mse = np.mean((original - marked) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

# --- Simulação e Carregamento de Imagem Original ---
def setup_image(filename='imagem_original.png', mode='L', size=(256, 256)):
    if not os.path.exists(filename):
        print(f"Criando uma imagem simulada '{filename}' ({mode}) para demonstração.")
        if mode == 'L':
            img_simulada = np.random.randint(0, 256, size=size, dtype=np.uint8)
        elif mode == 'RGB':
            img_simulada = np.random.randint(0, 256, size=(size[0], size[1], 3), dtype=np.uint8)
        Image.fromarray(img_simulada).save(filename)
    return Image.open(filename).convert(mode)

# --- Processo Principal ---
if __name__ == "__main__":
    # --- Parte 0: Preparação e Simulação de Imagem Original (Escala de Cinza) ---
    print("--- Configuração da Imagem em Escala de Cinza ---")
    img_gray = setup_image(mode='L')
    img_array_gray = np.array(img_gray)

    # Exemplo de criação de marca d'água aleatória
    marca_dim = (64, 64)
    marca_aleatoria = np.random.randint(0, 2, size=marca_dim, dtype=np.uint8)

    # Incorporação da marca d'água e salvamento
    img_watermarked_gray = embed_lsb(img_array_gray, marca_aleatoria, top=0, left=0)
    img_emb_gray = Image.fromarray(img_watermarked_gray)
    img_emb_gray.save('imagem_marcada.png')

    # Extração e exibição da marca d'água
    marca_extraida_gray = extract_lsb(img_watermarked_gray, marca_aleatoria.shape, top=0, left=0)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img_array_gray, cmap='gray')
    plt.title("Imagem Original (Cinza)")
    plt.subplot(1, 3, 2)
    plt.imshow(img_emb_gray, cmap='gray')
    plt.title("Imagem Marcada (Cinza)")
    plt.subplot(1, 3, 3)
    plt.imshow(marca_extraida_gray, cmap='gray')
    plt.title("Marca d'água Extraída (Original)")
    plt.tight_layout()
    plt.show()

    print("\n--- 1. Comparação de Qualidade (PSNR) ---")
    psnr_value = calculate_psnr(img_array_gray, img_watermarked_gray)
    print(f"PSNR entre a imagem original e a marcada: {psnr_value:.2f} dB")
    print("Um PSNR alto indica que a marca d'água é visualmente imperceptível.")

    # --- Parte 2: Testes de Robustez ---
    print("\n--- 2. Testes de Robustez ---")

    # Teste com Ruído Aleatório (Gaussiano)
    print("\n  a) Teste com Ruído Aleatório:")
    img_watermarked_noisy = img_watermarked_gray.copy().astype(np.float32)
    noise = np.random.normal(0, 5, img_watermarked_noisy.shape) # Desvio padrão 5 para ruído perceptível
    img_watermarked_noisy = np.clip((img_watermarked_noisy + noise), 0, 255).astype(np.uint8)

    marca_extraida_ruido = extract_lsb(img_watermarked_noisy, marca_aleatoria.shape, top=0, left=0)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img_array_gray, cmap='gray')
    plt.title("Imagem Original")
    plt.subplot(1, 3, 2)
    plt.imshow(img_watermarked_noisy, cmap='gray')
    plt.title("Imagem Marcada com Ruído")
    plt.subplot(1, 3, 3)
    plt.imshow(marca_extraida_ruido, cmap='gray')
    plt.title("Marca d'água Extraída (Ruído)")
    plt.tight_layout()
    plt.show()
    print("A marca d'água extraída após adicionar ruído deve estar visivelmente corrompida, destacando a fragilidade do LSB.")

    # Teste com Compressão JPEG
    print("\n  b) Teste com Compressão JPEG:")
    img_emb_jpeg = Image.fromarray(img_watermarked_gray)
    img_emb_jpeg.save('imagem_marcada_jpeg.jpg', quality=70) # Qualidade 70%

    img_reopened_jpeg = Image.open('imagem_marcada_jpeg.jpg').convert('L')
    img_reopened_jpeg_array = np.array(img_reopened_jpeg)

    marca_extraida_jpeg = extract_lsb(img_reopened_jpeg_array, marca_aleatoria.shape, top=0, left=0)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img_array_gray, cmap='gray')
    plt.title("Imagem Original")
    plt.subplot(1, 3, 2)
    plt.imshow(img_reopened_jpeg_array, cmap='gray')
    plt.title("Imagem Marcada (JPEG)")
    plt.subplot(1, 3, 3)
    plt.imshow(marca_extraida_jpeg, cmap='gray')
    plt.title("Marca d'água Extraída (JPEG)")
    plt.tight_layout()
    plt.show()
    print("A marca d'água extraída após compressão JPEG deve estar severamente corrompida, demonstrando a baixa robustez do LSB a este tipo de ataque.")

    # --- Parte 3: Esconder uma Pequena Mensagem Textual ---
    print("\n--- 3. Esconder uma Mensagem Textual ---")
    mensagem_secreta = "Olá, LSB é legal! - Recife, PE"
    bits_mensagem = text_to_bits(mensagem_secreta)

    num_pixels_necessarios = len(bits_mensagem)
    marca_h_msg = int(np.ceil(np.sqrt(num_pixels_necessarios)))
    marca_w_msg = marca_h_msg

    if marca_h_msg * marca_w_msg < num_pixels_necessarios:
        marca_w_msg += 1

    marca_mensagem = np.zeros((marca_h_msg, marca_w_msg), dtype=np.uint8)
    for i in range(num_pixels_necessarios):
        row = i // marca_w_msg
        col = i % marca_w_msg
        if row < marca_h_msg and col < marca_w_msg:
            marca_mensagem[row, col] = bits_mensagem[i]

    img_watermarked_text = embed_lsb(img_array_gray, marca_mensagem, top=0, left=0)
    img_emb_text = Image.fromarray(img_watermarked_text)
    img_emb_text.save('imagem_com_mensagem.png')

    marca_extraida_text_bits_raw = extract_lsb(img_watermarked_text, marca_mensagem.shape, top=0, left=0)
    extraido_bits_flat = marca_extraida_text_bits_raw.flatten()
    extraido_bits_mensagem = extraido_bits_flat[:len(bits_mensagem)]

    mensagem_extraida = bits_to_text(extraido_bits_mensagem)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_array_gray, cmap='gray')
    plt.title("Imagem Original")
    plt.subplot(1, 2, 2)
    plt.imshow(img_emb_text, cmap='gray')
    plt.title("Imagem com Mensagem Secreta")
    plt.tight_layout()
    plt.show()

    print(f"Mensagem Original: '{mensagem_secreta}'")
    print(f"Mensagem Extraída: '{mensagem_extraida}'")
    if mensagem_secreta == mensagem_extraida:
        print("A mensagem foi incorporada e extraída com sucesso!")
    else:
        print("Houve um erro na incorporação ou extração da mensagem.")

    # --- Parte 4: Modificação para Imagens Coloridas (RGB) ---
    print("\n--- 4. Modificação para Imagens Coloridas (RGB) ---")
    img_rgb = setup_image(mode='RGB')
    img_array_rgb = np.array(img_rgb)

    # Reutilizando a marca d'água aleatória para demonstração em RGB
    img_watermarked_rgb = embed_lsb_rgb(img_array_rgb, marca_aleatoria, top=0, left=0, channel=2) # Canal Azul
    img_emb_rgb = Image.fromarray(img_watermarked_rgb)
    img_emb_rgb.save('imagem_marcada_rgb.png')

    marca_extraida_rgb = extract_lsb_rgb(img_watermarked_rgb, marca_aleatoria.shape, top=0, left=0, channel=2)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img_array_rgb)
    plt.title("Imagem Original RGB")
    plt.subplot(1, 3, 2)
    plt.imshow(img_emb_rgb)
    plt.title("Imagem Marcada RGB")
    plt.subplot(1, 3, 3)
    plt.imshow(marca_extraida_rgb, cmap='gray')
    plt.title("Marca d'água Extraída RGB")
    plt.tight_layout()
    plt.show()

    if np.array_equal(marca_aleatoria, marca_extraida_rgb):
        print("Marca d'água RGB incorporada e extraída com sucesso no canal azul!")
    else:
        print("Erro na incorporação ou extração da marca d'água RGB.")

    # --- Limpeza de arquivos gerados ---
    print("\n--- Limpando arquivos gerados para nova execução ---")
    files_to_remove = ['imagem_original.png', 'imagem_marcada.png',
                       'imagem_marcada_jpeg.jpg', 'imagem_com_mensagem.png',
                       'imagem_marcada_rgb.png']
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removido: {f}")