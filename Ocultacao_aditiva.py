from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score
import cv2  # Para ruído sal e pimenta

# --- Funções Auxiliares ---

def calculate_psnr(original, marked):
    """Calcula o Peak Signal-to-Noise Ratio (PSNR) entre duas imagens."""
    mse = np.mean((original - marked) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def text_to_binary(text_message):
    """Converte uma mensagem de texto em uma sequência de bits (0s e 1s)."""
    binary_message = ''.join(format(ord(char), '08b') for char in text_message)
    return np.array([int(bit) for bit in binary_message])

def binary_to_text(binary_array):
    """Converte uma sequência de bits (0s e 1s) de volta para texto."""
    if len(binary_array) % 8 != 0:
        # Pad with zeros if not a multiple of 8 (e.g., if extracted message is truncated)
        binary_array = np.pad(binary_array, (0, 8 - (len(binary_array) % 8)), 'constant', constant_values=0)

    binary_string = ''.join(str(bit) for bit in binary_array)
    text_message = ""
    for i in range(0, len(binary_string), 8):
        byte = binary_string[i:i+8]
        text_message += chr(int(byte, 2))
    return text_message

def add_salt_and_pepper_noise(image, amount=0.01):
    """Adiciona ruído sal e pimenta a uma imagem."""
    row, col = image.shape
    s_vs_p = 0.5
    amount = amount
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0
    return out

# --- Funções de Marca D'água ---

def embed_additive(image, watermark, alpha, top=0, left=0, channel=None):
    """
    Incorpora uma marca d'água em uma imagem usando o método aditivo.
    Para imagens coloridas, especifique o canal (0, 1 ou 2 para R, G, B).
    """
    img_copy = image.astype(np.float32).copy()
    h, w = watermark.shape

    if img_copy.ndim == 3 and channel is not None:  # Imagem colorida
        for i in range(h):
            for j in range(w):
                pixel = img_copy[top + i, left + j, channel]
                pixel_marked = pixel + alpha * watermark[i, j]
                img_copy[top + i, left + j, channel] = np.clip(pixel_marked, 0, 255)
    elif img_copy.ndim == 2 and channel is None:  # Imagem em escala de cinza
        for i in range(h):
            for j in range(w):
                pixel = img_copy[top + i, left + j]
                pixel_marked = pixel + alpha * watermark[i, j]
                img_copy[top + i, left + j] = np.clip(pixel_marked, 0, 255)
    else:
        raise ValueError("Combinação inválida de imagem e canal. Para RGB, especifique 'channel'. Para Grayscale, 'channel' deve ser None.")

    return img_copy.astype(np.uint8)


def extract_additive(original, marked, marca_shape, alpha, top=0, left=0, channel=None):
    """
    Extrai uma marca d'água de uma imagem usando o método aditivo.
    Para imagens coloridas, especifique o canal (0, 1 ou 2 para R, G, B).
    """
    h, w = marca_shape
    extracted = np.zeros((h, w), dtype=np.uint8)

    if original.ndim == 3 and channel is not None:  # Imagem colorida
        for i in range(h):
            for j in range(w):
                diff = marked[top + i, left + j, channel] - original[top + i, left + j, channel]
                w_hat = 1 if diff >= 0 else -1
                extracted[i, j] = 1 if w_hat == 1 else 0
    elif original.ndim == 2 and channel is None:  # Imagem em escala de cinza
        for i in range(h):
            for j in range(w):
                diff = marked[top + i, left + j] - original[top + i, left + j]
                w_hat = 1 if diff >= 0 else -1
                extracted[i, j] = 1 if w_hat == 1 else 0
    else:
        raise ValueError("Combinação inválida de imagem e canal. Para RGB, especifique 'channel'. Para Grayscale, 'channel' deve ser None.")

    return extracted


# --- Configurações Iniciais ---
if not os.path.exists("resultados"):
    os.makedirs("resultados")

# Carrega a imagem original (será convertida para L ou mantida RGB dependendo do teste)
img_original_path = 'imagem_original.png' # Certifique-se de ter esta imagem no mesmo diretório
try:
    img_original_gray = Image.open(img_original_path).convert('L')
    img_original_rgb = Image.open(img_original_path).convert('RGB')
except FileNotFoundError:
    print(f"Erro: O arquivo '{img_original_path}' não foi encontrado.")
    print("Por favor, crie um arquivo 'imagem_original.png' no mesmo diretório.")
    exit()

img_array_gray = np.array(img_original_gray)
img_array_rgb = np.array(img_original_rgb)

# --- Marca D'água Binária (exemplo) ---
# Você pode usar uma marca d'água aleatória ou uma convertida de texto
marca_shape_default = (64, 64)
# marca_binaria_aleatoria = np.random.randint(0, 2, size=marca_shape_default, dtype=np.uint8)
# marca_default = 2 * marca_binaria_aleatoria - 1

# Exemplo: Usando uma mensagem textual como marca d'água
mensagem_secreta = "Digital Watermarking is Fun!"
marca_binaria_text = text_to_binary(mensagem_secreta)
# Redimensiona a marca de texto para se ajustar a uma forma quadrada (aproximadamente)
side_length = int(np.sqrt(len(marca_binaria_text)))
marca_binaria_text_reshaped = marca_binaria_text[:side_length * side_length].reshape((side_length, side_length))
marca_default = 2 * marca_binaria_text_reshaped - 1

print(f"Dimensões da marca d'água: {marca_default.shape}")


# --- Parâmetros ---
# Valores de alpha para experimentar
alpha_values = [5, 10, 20] # Força da marca d'água
top_pos = 0
left_pos = 0

# --- Teste 1: Impacto de Alpha em Imagens em Escala de Cinza e PSNR ---
print("\n--- Teste 1: Impacto de Alpha em Imagens em Escala de Cinza e PSNR ---")
results_alpha_gray = {}

for alpha in alpha_values:
    print(f"\nTestando com alpha = {alpha} (Grayscale)")
    img_watermarked_gray = embed_additive(img_array_gray, marca_default, alpha=alpha, top=top_pos, left=left_pos)
    Image.fromarray(img_watermarked_gray).save(f'resultados/imagem_marcada_gray_alpha_{alpha}.png')

    marca_extraida_gray = extract_additive(img_array_gray, img_watermarked_gray, marca_default.shape, alpha=alpha, top=top_pos, left=left_pos)

    psnr_value = calculate_psnr(img_array_gray, img_watermarked_gray)
    accuracy = accuracy_score((marca_default + 1) // 2, marca_extraida_gray) # Converte de {-1,1} para {0,1} para comparação

    results_alpha_gray[alpha] = {'psnr': psnr_value, 'accuracy': accuracy}

    print(f"PSNR (Grayscale, alpha={alpha}): {psnr_value:.2f} dB")
    print(f"Acurácia (Grayscale, alpha={alpha}): {accuracy:.4f}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_array_gray, cmap='gray')
    plt.title("Original (Gray)")
    plt.subplot(1, 3, 2)
    plt.imshow(img_watermarked_gray, cmap='gray')
    plt.title(f"Marcada (Gray, Alpha={alpha})")
    plt.subplot(1, 3, 3)
    plt.imshow(marca_extraida_gray, cmap='gray')
    plt.title(f"Marca Extraída (Gray, Alpha={alpha})")
    plt.savefig(f'resultados/visualizacao_gray_alpha_{alpha}.png')
    plt.close()

# --- Teste 2: Robustez contra Compressão JPEG e Ruído Sal e Pimenta (Grayscale) ---
print("\n--- Teste 2: Robustez contra Compressão JPEG e Ruído Sal e Pimenta (Grayscale) ---")
alpha_robustness = 10 # Um valor de alpha médio para este teste

img_watermarked_base = embed_additive(img_array_gray, marca_default, alpha=alpha_robustness, top=top_pos, left=left_pos)
Image.fromarray(img_watermarked_base).save('resultados/imagem_marcada_base_robustez.png')

# 2.1 Compressão JPEG
print("\nTestando Robustez: Compressão JPEG")
jpeg_qualities = [90, 50, 10] # Qualidade JPEG

for quality in jpeg_qualities:
    img_pil_marked = Image.fromarray(img_watermarked_base)
    img_pil_marked.save(f'resultados/imagem_marcada_jpeg_q{quality}.jpg', quality=quality)
    img_compressed = Image.open(f'resultados/imagem_marcada_jpeg_q{quality}.jpg').convert('L')
    img_compressed_array = np.array(img_compressed)

    marca_extraida_jpeg = extract_additive(img_array_gray, img_compressed_array, marca_default.shape, alpha=alpha_robustness, top=top_pos, left=left_pos)
    accuracy_jpeg = accuracy_score((marca_default + 1) // 2, marca_extraida_jpeg)

    print(f"Acurácia após JPEG Q{quality}: {accuracy_jpeg:.4f}")

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_compressed_array, cmap='gray')
    plt.title(f"Marcada + JPEG Q{quality}")
    plt.subplot(1, 2, 2)
    plt.imshow(marca_extraida_jpeg, cmap='gray')
    plt.title(f"Marca Extraída (JPEG Q{quality})")
    plt.savefig(f'resultados/visualizacao_jpeg_q{quality}.png')
    plt.close()

# 2.2 Ruído Sal e Pimenta
print("\nTestando Robustez: Ruído Sal e Pimenta")
noise_amounts = [0.001, 0.01, 0.05] # Quantidade de ruído

for amount in noise_amounts:
    img_noisy = add_salt_and_pepper_noise(img_watermarked_base, amount=amount)
    Image.fromarray(img_noisy).save(f'resultados/imagem_marcada_noise_sp_{amount}.png')

    marca_extraida_noise = extract_additive(img_array_gray, img_noisy, marca_default.shape, alpha=alpha_robustness, top=top_pos, left=left_pos)
    accuracy_noise = accuracy_score((marca_default + 1) // 2, marca_extraida_noise)

    print(f"Acurácia após Ruído S&P (amount={amount}): {accuracy_noise:.4f}")

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_noisy, cmap='gray')
    plt.title(f"Marcada + Ruído S&P ({amount})")
    plt.subplot(1, 2, 2)
    plt.imshow(marca_extraida_noise, cmap='gray')
    plt.title(f"Marca Extraída (Ruído S&P {amount})")
    plt.savefig(f'resultados/visualizacao_noise_sp_{amount}.png')
    plt.close()

# --- Teste 3: Marca D'água em Imagens Coloridas (Canal RGB) ---
print("\n--- Teste 3: Marca D'água em Imagens Coloridas (Canal RGB) ---")
alpha_color = 10 # Força da marca d'água para imagens coloridas
channel_to_embed = 1 # Canal Verde (0=R, 1=G, 2=B)

print(f"Incorporando marca d'água no canal {channel_to_embed} (Green) da imagem RGB.")
img_watermarked_rgb = embed_additive(img_array_rgb, marca_default, alpha=alpha_color, top=top_pos, left=left_pos, channel=channel_to_embed)
Image.fromarray(img_watermarked_rgb).save(f'resultados/imagem_marcada_rgb_channel_{channel_to_embed}.png')

marca_extraida_rgb = extract_additive(img_array_rgb, img_watermarked_rgb, marca_default.shape, alpha=alpha_color, top=top_pos, left=left_pos, channel=channel_to_embed)
accuracy_rgb = accuracy_score((marca_default + 1) // 2, marca_extraida_rgb)

psnr_color = calculate_psnr(img_array_rgb, img_watermarked_rgb) # PSNR para imagem colorida

print(f"PSNR (RGB, Canal {channel_to_embed}): {psnr_color:.2f} dB")
print(f"Acurácia (RGB, Canal {channel_to_embed}): {accuracy_rgb:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img_array_rgb)
plt.title("Original (RGB)")
plt.subplot(1, 3, 2)
plt.imshow(img_watermarked_rgb)
plt.title(f"Marcada (RGB, Canal {channel_to_embed})")
plt.subplot(1, 3, 3)
plt.imshow(marca_extraida_rgb, cmap='gray')
plt.title(f"Marca Extraída (RGB, Canal {channel_to_embed})")
plt.savefig(f'resultados/visualizacao_rgb_channel_{channel_to_embed}.png')
plt.close()

# --- Teste 4: Inserção e Extração de Mensagem Textual ---
print("\n--- Teste 4: Inserção e Extração de Mensagem Textual ---")
mensagem_original = "Hello, world! This is a secret message for watermarking demonstration."
print(f"Mensagem original: '{mensagem_original}'")

marca_binaria_text_embed = text_to_binary(mensagem_original)

# Garantir que a marca d'água textual tenha um tamanho que possa ser reshapeado
# Para simplificar, vamos criar uma marca quadrada a partir da mensagem binária
side_length_text = int(np.sqrt(len(marca_binaria_text_embed)))
# Pad ou truncate a mensagem para caber em um quadrado perfeito
if len(marca_binaria_text_embed) < side_length_text * side_length_text:
    marca_binaria_text_embed = np.pad(marca_binaria_text_embed, (0, side_length_text * side_length_text - len(marca_binaria_text_embed)), 'constant', constant_values=0)
else:
    marca_binaria_text_embed = marca_binaria_text_embed[:side_length_text * side_length_text]

marca_text_embed = 2 * marca_binaria_text_embed.reshape((side_length_text, side_length_text)) - 1
print(f"Dimensões da marca textual para embedding: {marca_text_embed.shape}")

alpha_text = 15 # Alpha para mensagem textual

img_watermarked_text = embed_additive(img_array_gray, marca_text_embed, alpha=alpha_text, top=top_pos, left=left_pos)
Image.fromarray(img_watermarked_text).save('resultados/imagem_marcada_textual.png')

marca_extraida_text = extract_additive(img_array_gray, img_watermarked_text, marca_text_embed.shape, alpha=alpha_text, top=top_pos, left=left_pos)

# Converte a marca extraída de volta para texto
# A marca extraída está em {0, 1}, então não precisa converter de {-1, 1}
mensagem_extraida = binary_to_text(marca_extraida_text.flatten())
print(f"Mensagem extraída: '{mensagem_extraida}'")

# Comparação pixel a pixel da marca binária original e extraída
accuracy_text = accuracy_score((marca_text_embed + 1) // 2, marca_extraida_text)
print(f"Acurácia da extração da mensagem textual: {accuracy_text:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img_array_gray, cmap='gray')
plt.title("Original (Gray)")
plt.subplot(1, 3, 2)
plt.imshow(img_watermarked_text, cmap='gray')
plt.title(f"Marcada (Mensagem Textual)")
plt.subplot(1, 3, 3)
plt.imshow(marca_extraida_text, cmap='gray')
plt.title(f"Marca Textual Extraída")
plt.savefig('resultados/visualizacao_textual.png')
plt.close()


print("\n--- Relatório Final ---")
print("\nResultados do Teste 1 (Impacto de Alpha em Grayscale):")
for alpha, metrics in results_alpha_gray.items():
    print(f"Alpha={alpha}: PSNR={metrics['psnr']:.2f} dB, Acurácia={metrics['accuracy']:.4f}")

print("\nResultados do Teste 2 (Robustez - Grayscale, Alpha=10):")
print("  - Compressão JPEG:")
for quality in jpeg_qualities:
    img_compressed = Image.open(f'resultados/imagem_marcada_jpeg_q{quality}.jpg').convert('L')
    img_compressed_array = np.array(img_compressed)
    marca_extraida_jpeg = extract_additive(img_array_gray, img_compressed_array, marca_default.shape, alpha=alpha_robustness, top=top_pos, left=left_pos)
    accuracy_jpeg = accuracy_score((marca_default + 1) // 2, marca_extraida_jpeg)
    print(f"    JPEG Q{quality}: Acurácia={accuracy_jpeg:.4f}")

print("  - Ruído Sal e Pimenta:")
for amount in noise_amounts:
    img_noisy = Image.open(f'resultados/imagem_marcada_noise_sp_{amount}.png').convert('L')
    img_noisy_array = np.array(img_noisy)
    marca_extraida_noise = extract_additive(img_array_gray, img_noisy_array, marca_default.shape, alpha=alpha_robustness, top=top_pos, left=left_pos)
    accuracy_noise = accuracy_score((marca_default + 1) // 2, marca_extraida_noise)
    print(f"    Ruído S&P (amount={amount}): Acurácia={accuracy_noise:.4f}")

print("\nResultados do Teste 3 (RGB, Canal Verde):")
print(f"  PSNR={psnr_color:.2f} dB, Acurácia={accuracy_rgb:.4f}")

print("\nResultados do Teste 4 (Mensagem Textual):")
print(f"  Mensagem Original: '{mensagem_original}'")
print(f"  Mensagem Extraída: '{mensagem_extraida}'")
print(f"  Acurácia da Extração: {accuracy_text:.4f}")

print("\nTodos os resultados, incluindo imagens e gráficos, foram salvos na pasta 'resultados'.")
