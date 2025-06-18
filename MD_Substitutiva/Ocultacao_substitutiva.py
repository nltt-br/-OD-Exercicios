from PIL import Image # Importa a biblioteca Pillow para manipulação de imagens
import numpy as np # Importa NumPy para operações eficientes com arrays (pixels da imagem)
import matplotlib.pyplot as plt # Importa Matplotlib para exibir imagens e gráficos
from math import log10, sqrt # Importa funções matemáticas para o cálculo do PSNR (Peak Signal-to-Noise Ratio)
import os # Importa o módulo os para interagir com o sistema operacional, como verificar e remover arquivos

# --- Funções LSB (Escala de Cinza) ---
# Esta função incorpora uma marca d'água em uma imagem em escala de cinza usando o método LSB (Least Significant Bit).
def embed_lsb(image, watermark, top=0, left=0):
    """
    Incorpora uma marca d'água binária (0s e 1s) em uma imagem em escala de cinza.

    Args:
        image (numpy.ndarray): O array NumPy da imagem original em escala de cinza.
        watermark (numpy.ndarray): O array NumPy da marca d'água binária (0s e 1s).
        top (int): Coordenada Y (linha) de início para a incorporação da marca d'água.
        left (int): Coordenada X (coluna) de início para a incorporação da marca d'água.

    Returns:
        numpy.ndarray: Uma cópia da imagem com a marca d'água incorporada.
    """
    img_copy = image.copy() # Cria uma cópia da imagem para evitar modificar a original diretamente
    h, w = watermark.shape # Obtém as dimensões (altura e largura) da marca d'água

    # Itera sobre cada pixel da marca d'água
    for i in range(h):
        for j in range(w):
            # Verifica se as coordenadas da marca d'água estão dentro dos limites da imagem
            if top + i < img_copy.shape[0] and left + j < img_copy.shape[1]:
                pixel = img_copy[top+i, left+j] # Obtém o valor do pixel da imagem na posição atual
                # Modifica o LSB do pixel:
                # (pixel & 0xFE) - Zera o bit menos significativo do pixel (mantém os outros 7 bits)
                # | watermark[i, j] - Adiciona o bit da marca d'água ao LSB zerado
                pixel = (pixel & 0xFE) | watermark[i, j]
                img_copy[top+i, left+j] = pixel # Atualiza o pixel na imagem copiada
            # else:
            #     print(f"Aviso: Marca d'água excede os limites da imagem em ({top+i}, {left+j}).")
            # Este print foi comentado para evitar poluir o console, mas seria útil para depuração
    return img_copy # Retorna a imagem com a marca d'água

# Esta função extrai uma marca d'água de uma imagem em escala de cinza usando o método LSB.
def extract_lsb(image, marca_shape, top=0, left=0):
    """
    Extrai uma marca d'água binária (0s e 1s) de uma imagem em escala de cinza.

    Args:
        image (numpy.ndarray): O array NumPy da imagem de onde extrair a marca d'água.
        marca_shape (tuple): A forma (altura, largura) esperada da marca d'água.
        top (int): Coordenada Y (linha) de início para a extração da marca d'água.
        left (int): Coordenada X (coluna) de início para a extração da marca d'água.

    Returns:
        numpy.ndarray: O array NumPy da marca d'água extraída.
    """
    h, w = marca_shape # Obtém as dimensões esperadas da marca d'água
    extracted = np.zeros((h, w), dtype=np.uint8) # Cria um array de zeros para armazenar a marca d'água extraída

    # Itera sobre a área onde a marca d'água é esperada
    for i in range(h):
        for j in range(w):
            # Verifica se as coordenadas de extração estão dentro dos limites da imagem
            if top + i < image.shape[0] and left + j < image.shape[1]:
                # Extrai o LSB do pixel:
                # (image[top+i, left+j] & 1) - Aplica uma máscara para obter apenas o bit menos significativo
                extracted[i, j] = image[top+i, left+j] & 1
            # else:
            #     print(f"Aviso: Tentativa de extrair fora dos limites da imagem em ({top+i}, {left+j}).")
    return extracted # Retorna a marca d'água extraída

# --- Funções LSB (RGB) ---
# Esta função incorpora uma marca d'água em uma imagem RGB.
def embed_lsb_rgb(image_rgb, watermark, top=0, left=0, channel=2): # channel 0=R, 1=G, 2=B
    """
    Incorpora uma marca d'água binária (0s e 1s) em um canal específico de uma imagem RGB.

    Args:
        image_rgb (numpy.ndarray): O array NumPy da imagem RGB original.
        watermark (numpy.ndarray): O array NumPy da marca d'água binária (0s e 1s).
        top (int): Coordenada Y (linha) de início.
        left (int): Coordenada X (coluna) de início.
        channel (int): O canal de cor para incorporar (0 para Vermelho, 1 para Verde, 2 para Azul).

    Returns:
        numpy.ndarray: Uma cópia da imagem RGB com a marca d'água incorporada.
    """
    img_copy = image_rgb.copy() # Cria uma cópia da imagem RGB
    h, w = watermark.shape # Obtém as dimensões da marca d'água

    # Itera sobre cada pixel do watermark
    for i in range(h):
        for j in range(w):
            # Verifica se as coordenadas do watermark estão dentro dos limites da imagem
            if top + i < img_copy.shape[0] and left + j < img_copy.shape[1]:
                # Obtém o valor do pixel no canal de cor especificado
                pixel_channel = img_copy[top+i, left+j, channel]
                # Modifica o LSB do pixel no canal específico, assim como na função de escala de cinza
                pixel_channel = (pixel_channel & 0xFE) | watermark[i, j]
                img_copy[top+i, left+j, channel] = pixel_channel # Atualiza o pixel no canal
            # else:
            #     print(f"Aviso RGB: Watermark excede os limites da imagem em ({top+i}, {left+j}).")
    return img_copy # Retorna a imagem RGB com o watermark

# Esta função extrai uma marca d'água de um canal específico de uma imagem RGB.
def extract_lsb_rgb(image_rgb, marca_shape, top=0, left=0, channel=2): # channel 0=R, 1=G, 2=B
    """
    Extrai uma marca d'água binária (0s e 1s) de um canal específico de uma imagem RGB.

    Args:
        image_rgb (numpy.ndarray): O array NumPy da imagem RGB de onde extrair.
        marca_shape (tuple): A forma (altura, largura) esperada da marca d'água.
        top (int): Coordenada Y (linha) de início.
        left (int): Coordenada X (coluna) de início.
        channel (int): O canal de cor para extrair (0 para Vermelho, 1 para Verde, 2 para Azul).

    Returns:
        numpy.ndarray: O array NumPy da marca d'água extraída.
    """
    h, w = marca_shape # Obtém as dimensões esperadas da marca d'água
    extracted = np.zeros((h, w), dtype=np.uint8) # Cria um array de zeros para a marca d'água extraída

    # Itera sobre a área onde a marca d'água é esperada
    for i in range(h):
        for j in range(w):
            # Verifica se as coordenadas de extração estão dentro dos limites da imagem
            if top + i < image_rgb.shape[0] and left + j < image_rgb.shape[1]:
                # Extrai o LSB do pixel do canal especificado
                extracted[i, j] = image_rgb[top+i, left+j, channel] & 1
            # else:
            #     print(f"Aviso RGB: Tentativa de extrair fora dos limites da imagem em ({top+i}, {left+j}).")
    return extracted # Retorna a marca d'água extraída

# --- Funções de Conversão de Texto para Bits ---
# Converte uma string de texto em uma sequência de bits (0s e 1s).
def text_to_bits(text):
    """
    Converte uma string de texto em um array NumPy de bits (0s e 1s).

    Args:
        text (str): A string de texto a ser convertida.

    Returns:
        numpy.ndarray: Um array NumPy contendo os bits da mensagem.
    """
    bits = []
    for char in text:
        # Converte o caractere para seu valor ASCII, depois para binário,
        # remove o prefixo '0b' e preenche com zeros à esquerda para ter 8 bits.
        binary_char = bin(ord(char))[2:].zfill(8)
        for bit in binary_char:
            bits.append(int(bit)) # Adiciona cada bit individual à lista
    return np.array(bits, dtype=np.uint8) # Retorna como um array NumPy de tipo uint8

# Converte uma sequência de bits de volta para uma string de texto.
def bits_to_text(bits):
    """
    Converte um array NumPy de bits (0s e 1s) de volta para uma string de texto.

    Args:
        bits (numpy.ndarray): O array NumPy de bits a ser convertido.

    Returns:
        str: A string de texto reconstruída.
    """
    text = ""
    # Divide o array de bits em grupos de 8 (bytes)
    bytes_array = np.array_split(bits, len(bits) // 8)
    for byte_bits in bytes_array:
        if len(byte_bits) == 8: # Garante que temos um byte completo (8 bits)
            # Converte os 8 bits de volta para um inteiro e depois para um caractere ASCII
            byte_int = int("".join(str(b) for b in byte_bits), 2)
            text += chr(byte_int) # Adiciona o caractere à string resultante
    return text # Retorna a string de texto

# --- Função de Cálculo de PSNR ---
# Calcula o PSNR (Peak Signal-to-Noise Ratio) entre duas imagens.
def calculate_psnr(original, marked):
    """
    Calcula o PSNR (Peak Signal-to-Noise Ratio) entre duas imagens.

    Args:
        original (numpy.ndarray): O array NumPy da imagem original.
        marked (numpy.ndarray): O array NumPy da imagem marcada.

    Returns:
        float: O valor do PSNR em decibéis (dB), ou infinito se as imagens forem idênticas.
    """
    # Calcula o Mean Squared Error (MSE - Erro Quadrático Médio)
    mse = np.mean((original - marked) ** 2)
    if mse == 0: # Se o MSE for zero, as imagens são idênticas, o PSNR é infinito
        return float('inf')
    max_pixel = 255.0 # Valor máximo de pixel para imagens de 8 bits
    # Calcula o PSNR usando a fórmula
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

# --- Simulação e Carregamento de Imagem Original ---
# Função utilitária para configurar e carregar uma imagem, criando-a se não existir.
def setup_image(filename='imagem_original.png', mode='L', size=(256, 256)):
    """
    Configura e carrega uma imagem, criando-a com dados aleatórios se não existir.

    Args:
        filename (str): Nome do arquivo da imagem.
        mode (str): Modo da imagem ('L' para escala de cinza, 'RGB' para colorida).
        size (tuple): Dimensões da imagem (largura, altura).

    Returns:
        PIL.Image.Image: O objeto Image carregado.
    """
    if not os.path.exists(filename): # Verifica se o arquivo de imagem já existe
        print(f"Criando uma imagem simulada '{filename}' ({mode}) para demonstração.")
        if mode == 'L':
            # Cria um array NumPy com pixels aleatórios para imagem em escala de cinza
            img_simulada = np.random.randint(0, 256, size=size, dtype=np.uint8)
        elif mode == 'RGB':
            # Cria um array NumPy com pixels aleatórios para imagem RGB (3 canais)
            img_simulada = np.random.randint(0, 256, size=(size[0], size[1], 3), dtype=np.uint8)
        Image.fromarray(img_simulada).save(filename) # Salva o array como um arquivo de imagem
    return Image.open(filename).convert(mode) # Abre e retorna a imagem no modo especificado

# --- Bloco Principal de Execução ---
# Este bloco é executado apenas quando o script é executado diretamente (não quando importado como módulo).
if __name__ == "__main__":
    # --- Parte 0: Preparação e Simulação de Imagem Original (Escala de Cinza) ---
    print("--- Configuração da Imagem em Escala de Cinza ---")
    img_gray = setup_image(mode='L') # Carrega ou cria uma imagem em escala de cinza
    img_array_gray = np.array(img_gray) # Converte a imagem PIL para um array NumPy

    # Exemplo de criação de marca d'água aleatória
    marca_dim = (64, 64) # Define as dimensões da marca d'água
    marca_aleatoria = np.random.randint(0, 2, size=marca_dim, dtype=np.uint8) # Cria uma marca d'água binária aleatória

    # Incorporação da marca d'água e salvamento
    # A marca d'água é inserida na imagem em escala de cinza
    img_watermarked_gray = embed_lsb(img_array_gray, marca_aleatoria, top=0, left=0)
    img_emb_gray = Image.fromarray(img_watermarked_gray) # Converte o array NumPy de volta para imagem PIL
    img_emb_gray.save('imagem_marcada.png') # Salva a imagem marcada

    # Extração e exibição da marca d'água
    # A marca d'água é extraída da imagem marcada
    marca_extraida_gray = extract_lsb(img_watermarked_gray, marca_aleatoria.shape, top=0, left=0)

    # Exibe as imagens usando Matplotlib
    plt.figure(figsize=(12, 6)) # Cria uma figura com um tamanho específico
    plt.subplot(1, 3, 1) # Define uma grade de 1 linha, 3 colunas e seleciona a 1ª posição
    plt.imshow(img_array_gray, cmap='gray') # Exibe a imagem original em escala de cinza
    plt.title("Imagem Original (Cinza)") # Define o título do subplot
    plt.subplot(1, 3, 2) # Seleciona a 2ª posição
    plt.imshow(img_emb_gray, cmap='gray') # Exibe a imagem marcada
    plt.title("Imagem Marcada (Cinza)")
    plt.subplot(1, 3, 3) # Seleciona a 3ª posição
    plt.imshow(marca_extraida_gray, cmap='gray') # Exibe a marca d'água extraída
    plt.title("Marca d'água Extraída (Original)")
    plt.tight_layout() # Ajusta o layout para evitar sobreposição de títulos
    plt.show()

    # --- Análise da Parte 0 ---
    # Resultados: As imagens 'Original' e 'Marcada' são visualmente indistinguíveis,
    # demonstrando a imperceptibilidade do método LSB. A 'Marca d'água Extraída'
    # é idêntica à marca d'água original, confirmando a correta incorporação e extração
    # em um cenário ideal.
    # Dificuldades: Nenhuma nesta fase, o processo é direto e eficaz sem interferências externas.


    # --- 1. Comparação de Qualidade (PSNR) ---
    print("\n--- 1. Comparação de Qualidade (PSNR) ---")
    # Calcula o PSNR entre a imagem original e a marcada
    psnr_value = calculate_psnr(img_array_gray, img_watermarked_gray)
    print(f"PSNR entre a imagem original e a marcada: {psnr_value:.2f} dB")
    print("Um PSNR alto indica que a marca d'água é visualmente imperceptível.")

    # --- Análise da Parte 1 ---
    # Resultados: O PSNR será muito alto (geralmente acima de 40 dB ou 'inf'), o que reforça
    # que a alteração no LSB dos pixels é mínima e não causa degradação visual perceptível
    # na imagem marcada.
    # Dificuldades: Nenhuma. O cálculo é matemático e direto.


    # --- Parte 2: Testes de Robustez ---
    print("\n--- 2. Testes de Robustez ---")
    # A robustez mede a capacidade da marca d'água de resistir a ataques ou modificações.
    # O LSB é conhecido por sua baixa robustez.

    # Teste com Ruído Aleatório (Gaussiano)
    print("\n  a) Teste com Ruído Aleatório:")
    img_watermarked_noisy = img_watermarked_gray.copy().astype(np.float32) # Cria uma cópia para adicionar ruído
    # Gera ruído gaussiano (normal) com média 0 e desvio padrão 5
    noise = np.random.normal(0, 5, img_watermarked_noisy.shape)
    # Adiciona o ruído e garante que os valores dos pixels permaneçam entre 0 e 255
    img_watermarked_noisy = np.clip((img_watermarked_noisy + noise), 0, 255).astype(np.uint8)

    # Tenta extrair a marca d'água da imagem com ruído
    marca_extraida_ruido = extract_lsb(img_watermarked_noisy, marca_aleatoria.shape, top=0, left=0)

    # Exibe as imagens para comparação
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

    # --- Análise do Teste com Ruído ---
    # Resultados: A imagem da 'Marca d'água Extraída (Ruído)' aparece completamente
    # desorganizada e irreconhecível.
    # Dificuldades: A fragilidade do LSB é evidente aqui. Qualquer pequena alteração
    # nos valores dos pixels, como a adição de ruído aleatório (que modifica diretamente
    # os bits dos pixels), compromete drasticamente a marca d'água escondida. Isso ocorre
    # porque a marca d'água reside justamente nos bits mais vulneráveis.


    # Teste com Compressão JPEG
    print("\n  b) Teste com Compressão JPEG:")
    img_emb_jpeg = Image.fromarray(img_watermarked_gray) # Converte para imagem PIL
    # Salva a imagem como JPEG com qualidade 70 (compressão com perdas)
    img_emb_jpeg.save('imagem_marcada_jpeg.jpg', quality=70)

    # Reabre a imagem JPEG para simular a perda de dados da compressão
    img_reopened_jpeg = Image.open('imagem_marcada_jpeg.jpg').convert('L')
    img_reopened_jpeg_array = np.array(img_reopened_jpeg)

    # Tenta extrair a marca d'água da imagem JPEG
    marca_extraida_jpeg = extract_lsb(img_reopened_jpeg_array, marca_aleatoria.shape, top=0, left=0)

    # Exibe as imagens para comparação
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

    # --- Análise do Teste com Compressão JPEG ---
    # Resultados: A 'Marca d'água Extraída (JPEG)' também estará severamente corrompida,
    # semelhante ao resultado com ruído.
    # Dificuldades: A compressão JPEG é um processo com perdas que modifica os bits
    # dos pixels para reduzir o tamanho do arquivo. Essas modificações afetam diretamente
    # o LSB, destruindo a marca d'água. Isso mostra que o LSB não é robusto para
    # cenários onde a imagem será comprimida, um processo muito comum.


    # --- Parte 3: Esconder uma Pequena Mensagem Textual ---
    print("\n--- 3. Esconder uma Mensagem Textual ---")
    mensagem_secreta = "Olá, LSB é legal! - Recife, PE" # A mensagem de texto a ser escondida
    bits_mensagem = text_to_bits(mensagem_secreta) # Converte a mensagem em bits

    num_pixels_necessarios = len(bits_mensagem) # Número total de bits na mensagem
    # Calcula as dimensões (altura e largura) necessárias para a marca d'água
    # para acomodar todos os bits, tentando uma forma quase quadrada.
    marca_h_msg = int(np.ceil(np.sqrt(num_pixels_necessarios)))
    marca_w_msg = marca_h_msg

    # Ajusta a largura se a área calculada não for suficiente
    # (garante que todos os bits da mensagem caibam)
    if marca_h_msg * marca_w_msg < num_pixels_necessarios:
        marca_w_msg += 1

    # Cria um array de zeros para a marca d'água e preenche com os bits da mensagem
    marca_mensagem = np.zeros((marca_h_msg, marca_w_msg), dtype=np.uint8)
    for i in range(num_pixels_necessarios):
        row = i // marca_w_msg
        col = i % marca_w_msg
        if row < marca_h_msg and col < marca_w_msg: # Evita erros de índice
            marca_mensagem[row, col] = bits_mensagem[i]

    # Incorpora a marca d'água contendo a mensagem na imagem em escala de cinza
    img_watermarked_text = embed_lsb(img_array_gray, marca_mensagem, top=0, left=0)
    img_emb_text = Image.fromarray(img_watermarked_text)
    img_emb_text.save('imagem_com_mensagem.png') # Salva a imagem com a mensagem

    # Extrai a marca d'água da imagem com a mensagem
    marca_extraida_text_bits_raw = extract_lsb(img_watermarked_text, marca_mensagem.shape, top=0, left=0)
    extraido_bits_flat = marca_extraida_text_bits_raw.flatten() # Acha todos os bits extraídos
    # Pega apenas a quantidade de bits que corresponde à mensagem original
    extraido_bits_mensagem = extraido_bits_flat[:len(bits_mensagem)]

    mensagem_extraida = bits_to_text(extraido_bits_mensagem) # Converte os bits extraídos de volta para texto

    # Exibe a imagem original e a imagem com a mensagem
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

    # --- Análise da Parte 3 ---
    # Resultados: A mensagem é incorporada e extraída perfeitamente, e a imagem
    # com a mensagem é visualmente idêntica à original.
    # Dificuldades: A principal dificuldade é o gerenciamento do espaço necessário.
    # É preciso calcular a área da marca d'água para garantir que todos os bits da
    # mensagem caibam na imagem sem exceder seus limites ou causar detecção.
    # Para mensagens muito grandes, o LSB pode não ser a melhor opção, pois demandaria
    # muitas modificações no LSB, aumentando a chance de detecção ou artefatos.


    # --- Parte 4: Modificação para Imagens Coloridas (RGB) ---
    print("\n--- 4. Modificação para Imagens Coloridas (RGB) ---")
    img_rgb = setup_image(mode='RGB') # Carrega ou cria uma imagem RGB
    img_array_rgb = np.array(img_rgb) # Converte a imagem PIL para um array NumPy

    # Reutiliza a marca d'água aleatória para demonstração em RGB
    # A marca d'água é incorporada no canal azul (índice 2).
    # O olho humano é menos sensível a mudanças no canal azul e verde,
    # tornando essas escolhas mais discretas que o canal vermelho.
    img_watermarked_rgb = embed_lsb_rgb(img_array_rgb, marca_aleatoria, top=0, left=0, channel=2)
    img_emb_rgb = Image.fromarray(img_watermarked_rgb)
    img_emb_rgb.save('imagem_marcada_rgb.png') # Salva a imagem RGB marcada

    # Extrai a marca d'água do canal azul da imagem RGB marcada
    marca_extraida_rgb = extract_lsb_rgb(img_watermarked_rgb, marca_aleatoria.shape, top=0, left=0, channel=2)

    # Exibe as imagens RGB e a marca d'água extraída
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

    # Verifica se a marca d'água extraída é igual à original
    if np.array_equal(marca_aleatoria, marca_extraida_rgb):
        print("Marca d'água RGB incorporada e extraída com sucesso no canal azul!")
    else:
        print("Erro na incorporação ou extração da marca d'água RGB.")

    # --- Análise da Parte 4 ---
    # Resultados: O processo funciona de forma idêntica às imagens em escala de cinza,
    # com a imagem marcada sendo visualmente igual à original e a marca d'água
    # extraída corretamente.
    # Dificuldades: A principal consideração é a escolha do canal. A incorporação em
    # um canal específico (como o azul ou verde) tende a ser menos perceptível ao olho
    # humano do que no canal vermelho. O LSB em RGB pode, teoricamente, esconder
    # três vezes mais dados (um bit por canal), mas isso aumenta a chance de detecção
    # ou artefatos se todos os canais forem usados intensivamente.


    # --- Limpeza de arquivos gerados (descomente para ativar) ---
    # print("\n--- Limpando arquivos gerados para nova execução ---")
    # files_to_remove = ['imagem_original.png', 'imagem_marcada.png',
    #                    'imagem_marcada_jpeg.jpg', 'imagem_com_mensagem.png',
    #                    'imagem_marcada_rgb.png']
    # for f in files_to_remove:
    #     if os.path.exists(f): # Verifica se o arquivo existe antes de tentar remover
    #         os.remove(f) # Remove o arquivo
    #         print(f"Removido: {f}")