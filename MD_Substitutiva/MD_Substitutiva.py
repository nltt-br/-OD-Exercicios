from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

# --- Função para criar marca d'água de texto ASCII ---
def criar_marca_dagua_texto_ascii(texto, tamanho_fonte=10, marca_dim=(128, 64)):
    """
    Cria uma marca d'água em escala de cinza com texto ASCII.

    Args:
        texto (str): O texto a ser usado como marca d'água.
        tamanho_fonte (int): O tamanho da fonte para o texto.
        marca_dim (tuple): As dimensões (largura, altura) da marca d'água.

    Returns:
        numpy.ndarray: Uma matriz NumPy representando a marca d'água.
    """
    img = Image.new('L', marca_dim, color=0) # 'L' para tons de cinza, 0 para fundo preto
    desenho = ImageDraw.Draw(img)

    try:
        # Tenta carregar uma fonte padrão.
        # Você pode especificar o caminho para um arquivo .ttf se tiver uma fonte específica.
        fonte = ImageFont.truetype("arial.ttf", tamanho_fonte)
    except IOError:
        fonte = ImageFont.load_default()
        print("Aviso: Fonte 'arial.ttf' não encontrada, usando a fonte padrão do Pillow.")

    # Calcula a posição para centralizar o texto
    # textbbox retorna (left, top, right, bottom)
    bbox = desenho.textbbox((0, 0), texto, font=fonte)
    largura_texto = bbox[2] - bbox[0]
    altura_texto = bbox[3] - bbox[1]

    x = (marca_dim[0] - largura_texto) / 2
    y = (marca_dim[1] - altura_texto) / 2

    # Desenha o texto na imagem (cor branca, que é 255 em tons de cinza)
    desenho.text((x, y), texto, font=fonte, fill=255)

    # Converte a imagem para um array NumPy
    # A marca d'água LSB precisa ser binária (0 ou 1).
    # O texto desenhado em branco (255) e o fundo preto (0).
    # Dividindo por 255, transformamos 255 em 1 e 0 em 0.
    marca = np.array(img, dtype=np.uint8) // 255

    return marca

# --- Seu código principal de LSB ---
# Carregar a imagem original e converter para escala de cinza
try:
    img = Image.open('imagem_original.png').convert('L')
    img_array = np.array(img)
except FileNotFoundError:
    print("Erro: 'imagem_original.png' não encontrada. Certifique-se de que a imagem esteja no mesmo diretório.")
    exit()

# Gerar a marca d'água de texto ASCII
texto_marca_dagua = "SEGREDO"
marca_dimensoes = (150, 50) # Defina um tamanho adequado para sua marca d'água
marca = criar_marca_dagua_texto_ascii(texto_marca_dagua, tamanho_fonte=20, marca_dim=marca_dimensoes)

# --- Funções de Incorporação e Extração ---
# Otimizações: Mover o retorno do loop para fora da função
def embed_lsb(image, watermark, top=0, left=0):
    img_copy = image.copy()
    h, w = watermark.shape
    # Verifica se a marca d'água cabe na imagem
    if top + h > img_copy.shape[0] or left + w > img_copy.shape[1]:
        raise ValueError("Marca d'água muito grande para a posição especificada.")

    for i in range(h):
        for j in range(w):
            pixel = img_copy[top+i, left+j]
            # Limpa o LSB e então define-o com o bit da marca d'água
            img_copy[top+i, left+j] = (pixel & 0xFE) | watermark[i, j]
    return img_copy

def extract_lsb(image, marca_shape, top=0, left=0):
    h, w = marca_shape
    extracted = np.zeros((h, w), dtype=np.uint8)
    # Verifica se a área de extração está dentro dos limites da imagem
    if top + h > image.shape[0] or left + w > image.shape[1]:
        raise ValueError("Área de extração fora dos limites da imagem.")

    for i in range(h):
        for j in range(w):
            # Extrai o LSB do pixel
            extracted[i, j] = image[top+i, left+j] & 1
    return extracted

# --- Aplicar e Extrair a Marca D'água ---
img_watermarked = embed_lsb(img_array, marca, top=0, left=0)

# Salvar a imagem com marca d'água
img_emb = Image.fromarray(img_watermarked)
img_emb.save('imagem_marcada.png')
print("Imagem com marca d'água salva como 'imagem_marcada.png'")

# Extrair a marca d'água
marca_extraida = extract_lsb(img_watermarked, marca.shape, top=0, left=0)

# Exibir a marca d'água extraída
plt.imshow(marca_extraida, cmap='gray')
plt.title(f"Marca d'água extraída: '{texto_marca_dagua}'")
plt.show()