import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# Carrega a imagem
try:
    image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Certifique-se de que 'lena.png' está no mesmo diretório.")
except FileNotFoundError as e:
    print(e)
    # Cria uma imagem de teste caso 'lena.png' não seja encontrada
    image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    print("Usando imagem gerada aleatoriamente para demonstração.")


def block_dct(image, block_size=8):
    """
    Aplica a DCT em blocos da imagem.
    """
    h, w = image.shape
    dct_blocks = np.zeros_like(image, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            dct_blocks[i:i+block_size, j:j+block_size] = cv2.dct(np.float32(block))
    return dct_blocks

def embed_watermark(dct_image, watermark, alpha=0.05, block_size=8, positions=None):
    """
    Incorpora a marca d'água nos coeficientes DCT.

    Args:
        dct_image (np.array): Imagem no domínio DCT.
        watermark (list): A marca d'água (lista de bits 0 ou 1).
        alpha (float): Fator de intensidade da marca d'água.
        block_size (int): Tamanho do bloco.
        positions (list of tuples): Lista de tuplas (row, col) para as posições de inserção dentro de cada bloco.
                                     Se None, usa uma posição aleatória ou predefinida.
    """
    wm_index = 0
    watermarked = dct_image.copy()
    h, w = dct_image.shape

    # Posições padrão para inserção, se não forem fornecidas
    if positions is None:
        # Exemplo: usando coeficientes de média frequência
        # Você pode ajustar estas posições para ver o efeito
        positions = [(3, 3), (3, 4), (4, 3), (4, 4)]
        # Para posições aleatórias por bloco:
        # random_positions = [(random.randint(2, 5), random.randint(2, 5)) for _ in range((h // block_size) * (w // block_size))]


    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if wm_index >= len(watermark):
                break

            # Escolhe uma posição de inserção para o bloco atual
            # Se 'positions' for uma lista, itera sobre ela para cada bit da marca d'água
            # Ou, se quiser uma posição aleatória por bloco:
            # pos_r, pos_c = random.choice(positions) # para usar as posições aleatórias geradas acima
            
            # Usando uma posição fixa do array 'positions' para cada bit da marca d'água
            # Isso garante que cada bit seja inserido em uma posição específica, e as posições se repetem a cada 4 bits da marca d'água
            pos_r, pos_c = positions[wm_index % len(positions)]


            coeff = watermarked[i + pos_r, j + pos_c]
            # Modifica o coeficiente. Se watermark[wm_index] for 0, o coeficiente é levemente reduzido.
            # Se for 1, é levemente aumentado.
            watermarked[i + pos_r, j + pos_c] = coeff * (1 + alpha * (2 * watermark[wm_index] - 1))
            wm_index += 1
    return watermarked

def block_idct(dct_blocks, block_size=8):
    """
    Aplica a IDCT em blocos para reconstruir a imagem.
    """
    h, w = dct_blocks.shape
    image = np.zeros_like(dct_blocks, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_blocks[i:i+block_size, j:j+block_size]
            image[i:i+block_size, j:j+block_size] = cv2.idct(block)
    return np.uint8(np.clip(image, 0, 255))

def add_gaussian_noise(image, mean=0, std_dev=20):
    """
    Adiciona ruído gaussiano à imagem.
    """
    row, col = image.shape
    gauss = np.random.normal(mean, std_dev, (row, col))
    noisy_image = image + gauss
    return np.uint8(np.clip(noisy_image, 0, 255))

def extract_watermark(original_dct, watermarked_dct, alpha=0.05, block_size=8, watermark_length=None, positions=None):
    """
    Extrai a marca d'água baseada na relação entre os coeficientes DCT original e modificado.
    """
    extracted_wm = []
    h, w = original_dct.shape

    if positions is None:
        positions = [(3, 3), (3, 4), (4, 3), (4, 4)] # Posições padrão, devem ser as mesmas do embedding

    wm_count = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if watermark_length is not None and wm_count >= watermark_length:
                break

            pos_r, pos_c = positions[wm_count % len(positions)]

            original_coeff = original_dct[i + pos_r, j + pos_c]
            modified_coeff = watermarked_dct[i + pos_r, j + pos_c]

            # A lógica de extração é baseada na modificação:
            # modified_coeff = original_coeff * (1 + alpha * (2 * bit - 1))
            # (modified_coeff / original_coeff) - 1 = alpha * (2 * bit - 1)
            # ((modified_coeff / original_coeff) - 1) / alpha = 2 * bit - 1
            # 2 * bit = (((modified_coeff / original_coeff) - 1) / alpha) + 1
            # bit = ((((modified_coeff / original_coeff) - 1) / alpha) + 1) / 2

            if original_coeff != 0: # Evita divisão por zero
                ratio = (modified_coeff / original_coeff) - 1
                estimated_value = ratio / alpha

                # Decisão: se o valor estimado for mais próximo de -1 (para bit 0) ou 1 (para bit 1)
                # Adicione uma tolerância para robustez
                if estimated_value < 0: # Idealmente seria -1
                    extracted_wm.append(0)
                else: # Idealmente seria 1
                    extracted_wm.append(1)
            else:
                # Caso o coeficiente original seja zero, a extração pode ser ambígua.
                # Aqui, para simplificar, assumimos que não haverá modificação significativa
                # ou que o bit não pode ser extraído confiavelmente.
                # Você pode adicionar uma lógica mais robusta aqui.
                extracted_wm.append(0) # Ou alguma outra abordagem padrão

            wm_count += 1
            if watermark_length is not None and wm_count >= watermark_length:
                break
        if watermark_length is not None and wm_count >= watermark_length:
            break

    # Retorna apenas a quantidade de bits da marca d'água original
    return extracted_wm[:watermark_length] if watermark_length is not None else extracted_wm


# --- Testando as funcionalidades ---

wm = [1, 0, 1, 1, 0, 0, 1, 0] # Exemplo de marca d'água
# Posições para inserção (coeficientes de média frequência)
# Você pode experimentar com diferentes posições, por exemplo, mais próximos do DC (menores índices)
# ou mais distantes (maiores índices)
insertion_positions = [(2, 2), (3, 3), (4, 4), (5, 5)]


print(f"Marca d'água original: {wm}")

# 1. Testando diferentes posições de inserção
print("\n--- Testando diferentes posições de inserção ---")
dct_blocks_pos_test = block_dct(image)
dct_with_wm_pos_test = embed_watermark(dct_blocks_pos_test, wm, alpha=0.05, positions=insertion_positions)
image_with_wm_pos_test = block_idct(dct_with_wm_pos_test)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Imagem Original")
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title(f"Imagem com Marca d'Água (Posições: {insertion_positions})")
plt.imshow(image_with_wm_pos_test, cmap='gray')
plt.show()

# 2. Testando diferentes intensidades de alpha
print("\n--- Testando diferentes intensidades de alpha ---")
alphas = [0.01, 0.05, 0.1]
fig, axes = plt.subplots(1, len(alphas) + 1, figsize=(18, 6))
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original")
axes[0].axis('off')

for i, alpha_val in enumerate(alphas):
    dct_blocks_alpha = block_dct(image)
    dct_with_wm_alpha = embed_watermark(dct_blocks_alpha, wm, alpha=alpha_val, positions=insertion_positions)
    image_with_wm_alpha = block_idct(dct_with_wm_alpha)

    axes[i+1].imshow(image_with_wm_alpha, cmap='gray')
    axes[i+1].set_title(f"Alpha: {alpha_val}")
    axes[i+1].axis('off')
plt.tight_layout()
plt.show()

print("\n--- Avaliação da Perceptibilidade ---")
print("Observe as imagens geradas acima. Para um alpha de 0.01, a marca d'água é quase imperceptível.")
print("À medida que alpha aumenta para 0.05 e 0.1, a marca d'água pode se tornar ligeiramente visível,")
print("especialmente em áreas de baixo detalhe da imagem, como regiões planas.")
print("O objetivo é encontrar um equilíbrio entre perceptibilidade e robustez.")

# 3. Simulando um ataque simples (adição de ruído)
print("\n--- Simulando ataque de Ruído e Verificando Permanência ---")
dct_blocks_attack = block_dct(image)
dct_with_wm_attack = embed_watermark(dct_blocks_attack, wm, alpha=0.05, positions=insertion_positions)
image_with_wm_attack = block_idct(dct_with_wm_attack)

# Adicionando ruído à imagem com marca d'água
noisy_image_with_wm = add_gaussian_noise(image_with_wm_attack, mean=0, std_dev=30)

# Para verificar a permanência, precisamos da imagem atacada no domínio DCT
dct_noisy_image_with_wm = block_dct(noisy_image_with_wm)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Imagem com Marca d'Água")
plt.imshow(image_with_wm_attack, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Imagem com Marca d'Água + Ruído")
plt.imshow(noisy_image_with_wm, cmap='gray')
plt.show()

# 4. Implementando o extrator da marca
print("\n--- Extração da Marca d'Água ---")
# Extraindo da imagem original (sem ataque)
extracted_wm_original = extract_watermark(dct_blocks_pos_test, dct_with_wm_pos_test, alpha=0.05, watermark_length=len(wm), positions=insertion_positions)
print(f"Marca d'água extraída (sem ataque): {extracted_wm_original}")
print(f"Original: {wm}")
print(f"Correta: {extracted_wm_original == wm}")

# Extraindo da imagem atacada
extracted_wm_attacked = extract_watermark(dct_blocks_attack, dct_noisy_image_with_wm, alpha=0.05, watermark_length=len(wm), positions=insertion_positions)
print(f"Marca d'água extraída (após ruído): {extracted_wm_attacked}")
print(f"Original: {wm}")
print(f"Correta (após ruído): {extracted_wm_attacked == wm}")

# Calculando a taxa de erro de bit (BER)
def calculate_ber(original_wm, extracted_wm):
    if not original_wm or not extracted_wm:
        return 1.0 # Evita divisão por zero se uma das listas estiver vazia
    min_len = min(len(original_wm), len(extracted_wm))
    errors = sum(1 for i in range(min_len) if original_wm[i] != extracted_wm[i])
    return errors / min_len

ber_original = calculate_ber(wm, extracted_wm_original)
ber_attacked = calculate_ber(wm, extracted_wm_attacked)

print(f"Taxa de Erro de Bit (BER) (sem ataque): {ber_original:.4f}")
print(f"Taxa de Erro de Bit (BER) (após ruído): {ber_attacked:.4f}")