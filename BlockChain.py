import hashlib # Importa a biblioteca para funções de hash criptográfico (SHA-256)
import json # Importa a biblioteca para trabalhar com dados JSON (serialização e desserialização)
from datetime import datetime # Importa a classe datetime para trabalhar com carimbos de tempo (timestamps)
from colorama import Fore, Style, init # Importa classes do Colorama para estilizar a saída no terminal (cores)

# Inicializa o Colorama. O 'autoreset=True' garante que as cores sejam resetadas automaticamente após cada print.
init(autoreset=True)

class Block:
    """
    Representa um bloco individual na blockchain. Cada bloco contém dados de transação,
    um carimbo de tempo, um hash do bloco anterior, um nonce (para prova de trabalho)
    e seu próprio hash.
    """
    def __init__(self, index, timestamp, data, previous_hash, nonce=0):
        """
        Construtor para a classe Block.

        Args:
            index (int): O número de identificação único do bloco na cadeia.
            timestamp (str): O carimbo de tempo de quando o bloco foi criado.
            data (dict/str): Os dados (transações) contidos neste bloco.
            previous_hash (str): O hash do bloco anterior na cadeia, garantindo o encadeamento.
            nonce (int): Um número arbitrário usado na mineração para encontrar um hash válido.
        """
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = nonce # O nonce é usado na "mineração" para alterar o hash do bloco
        self.hash = self.calculate_hash() # O hash do bloco é calculado no momento da criação

    def calculate_hash(self):
        """
        Calcula o hash SHA-256 do bloco.

        Ele pega todos os atributos do bloco, serializa-os em uma string JSON ordenada,
        codifica essa string para bytes e então calcula o hash SHA-256.
        A ordenação das chaves (sort_keys=True) garante que o hash seja consistente.

        Returns:
            str: O hash SHA-256 hexadecimal do bloco.
        """
        # Cria uma string JSON a partir dos atributos do bloco.
        # sort_keys=True garante uma ordem consistente para o hash.
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }, sort_keys=True).encode() # .encode() converte a string para bytes
        return hashlib.sha256(block_string).hexdigest() # Retorna o hash em formato hexadecimal

    def __str__(self):
        """
        Método especial que define a representação em string do objeto Block.
        Permite que você use `print(block_instance)` para obter uma saída formatada e colorida.

        Returns:
            str: Uma string formatada contendo os detalhes do bloco.
        """
        return (
            # Cabeçalho do bloco com cor ciano
            f"{Fore.CYAN}--- Bloco {self.index} ---{Style.RESET_ALL}\n"
            # Detalhes do bloco com labels em amarelo e valores em cores apropriadas
            f"  {Fore.YELLOW}Timestamp:{Style.RESET_ALL} {self.timestamp}\n"
            f"  {Fore.YELLOW}Data:{Style.RESET_ALL} {self.data}\n"
            f"  {Fore.YELLOW}Previous Hash:{Style.RESET_ALL} {Fore.BLUE}{self.previous_hash}{Style.RESET_ALL}\n"
            f"  {Fore.YELLOW}Hash:{Style.RESET_ALL} {Fore.GREEN}{self.hash}{Style.RESET_ALL}\n"
        )


class Blockchain:
    """
    Representa a cadeia de blocos em si. Gerencia a adição de novos blocos,
    a criação do bloco gênese e a validação da integridade da cadeia.
    """
    def __init__(self):
        """
        Construtor para a classe Blockchain.

        Inicializa a cadeia com o bloco gênese (o primeiro bloco da blockchain).
        Também define um nível de dificuldade inicial para a mineração (se ativada).
        """
        self.chain = [self.create_genesis_block()] # A cadeia é uma lista de blocos, começando com o gênese
        self.difficulty = 2 # Nível de dificuldade para a prova de trabalho (número de zeros no início do hash)

    def create_genesis_block(self):
        """
        Cria o primeiro bloco da blockchain, conhecido como Bloco Gênese.
        Ele tem índice 0, um carimbo de tempo atual, dados "Genesis Block" e um previous_hash de "0".

        Returns:
            Block: O bloco gênese recém-criado.
        """
        return Block(0, str(datetime.now()), "Genesis Block", "0")

    def get_latest_block(self):
        """
        Retorna o último bloco adicionado à cadeia.

        Returns:
            Block: O bloco mais recente na blockchain.
        """
        return self.chain[-1]

    def add_block(self, new_data):
        """
        Adiciona um novo bloco à blockchain.

        Args:
            new_data (dict/str): Os dados (transações) a serem incluídos no novo bloco.
        """
        previous_block = self.get_latest_block() # Pega o último bloco para obter seu hash e índice
        new_block = Block(
            index=previous_block.index + 1, # O índice do novo bloco é o do anterior + 1
            timestamp=str(datetime.now()), # Carimbo de tempo atual
            data=new_data, # Dados fornecidos para o novo bloco
            previous_hash=previous_block.hash # O hash do bloco anterior é crucial para a segurança da cadeia
        )
        # Opcional: Descomente a linha abaixo para habilitar a prova de trabalho (mineração)
        # self.mine_block(new_block) # Tenta minerar o novo bloco para atender à dificuldade
        self.chain.append(new_block) # Adiciona o bloco minerado (ou não minerado, se a linha acima estiver comentada) à cadeia

    def mine_block(self, block):
        """
        Realiza a "mineração" de um bloco para encontrar um hash que atenda à dificuldade.
        Isso é feito incrementando o 'nonce' até que o hash do bloco comece com o número
        desejado de zeros (definido por self.difficulty).

        Args:
            block (Block): O bloco a ser minerado.
        """
        target_prefix = '0' * self.difficulty # Define o prefixo alvo, ex: '00' para difficulty=2
        while not block.hash.startswith(target_prefix): # Continua enquanto o hash não começar com o prefixo
            block.nonce += 1 # Incrementa o nonce
            block.hash = block.calculate_hash() # Recalcula o hash com o novo nonce
        # Imprime uma mensagem quando o bloco é minerado com sucesso
        print(f"{Fore.MAGENTA}Bloco {block.index} minerado com sucesso! Nonce: {block.nonce}{Style.RESET_ALL}")


    def is_chain_valid(self):
        """
        Verifica a integridade de toda a blockchain.

        Percorre cada bloco (a partir do segundo) e verifica duas condições:
        1. Se o hash armazenado no bloco é realmente o hash calculado a partir de seus dados.
        2. Se o 'previous_hash' do bloco atual corresponde ao hash do bloco anterior na cadeia.

        Returns:
            bool: True se a blockchain for válida, False caso contrário.
        """
        print(f"\n{Fore.YELLOW}--- Verificando Integridade da Blockchain ---{Style.RESET_ALL}")
        # Começa do segundo bloco (índice 1), pois o primeiro (gênese) não tem um "anterior" para verificar
        for i in range(1, len(self.chain)):
            current = self.chain[i] # O bloco atual
            previous = self.chain[i - 1] # O bloco anterior

            # 1. Verifica se o hash do bloco atual é consistente com seus próprios dados
            if current.hash != current.calculate_hash():
                print(f"{Fore.RED}[ERRO] Hash inválido no bloco {i}!{Style.RESET_ALL}")
                print(f"  Hash Esperado: {current.calculate_hash()}")
                print(f"  Hash Armazenado: {current.hash}")
                return False # Se o hash está incorreto, a cadeia é inválida
            
            # 2. Verifica se o 'previous_hash' do bloco atual corresponde ao hash do bloco anterior
            if current.previous_hash != previous.hash:
                print(f"{Fore.RED}[ERRO] Encadeamento quebrado entre os blocos {i-1} e {i}!{Style.RESET_ALL}")
                print(f"  Previous Hash do Bloco {i}: {current.previous_hash}")
                print(f"  Hash do Bloco {i-1}: {previous.hash}")
                return False # Se o encadeamento está quebrado, a cadeia é inválida
        
        print(f"{Fore.GREEN}Blockchain é válida!{Style.RESET_ALL}") # Se todas as verificações passarem
        return True

    def print_chain(self):
        """
        Imprime todos os blocos da blockchain de forma formatada e colorida.
        Utiliza o método __str__ da classe Block para cada bloco.
        """
        # Cabeçalhos para a impressão da blockchain
        print(f"\n{Fore.CYAN}{'='*40}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}--- Blockchain ({len(self.chain)} Blocos) ---{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*40}{Style.RESET_ALL}\n")
        
        # Itera sobre cada bloco na cadeia e o imprime
        for block in self.chain:
            print(block) # Isso chama o método __str__ do objeto Block
            print(f"{Fore.CYAN}{'-'*30}{Style.RESET_ALL}\n") # Separador visual entre os blocos


# --- Execução Principal do Código ---

# Criar uma nova instância da blockchain
my_chain = Blockchain()

# Adicionar blocos com dados de "transações" à blockchain
my_chain.add_block({"from": "Alice", "to": "Bob", "amount": 50})
my_chain.add_block({"from": "Bob", "to": "Charlie", "amount": 25})
my_chain.add_block({"from": "Charlie", "to": "Alice", "amount": 10})

# Imprimir a blockchain inicial para visualização
my_chain.print_chain()

# Verificar a integridade da blockchain recém-criada
my_chain.is_chain_valid()

# --- Simulação de Ataque ---
# Demonstra como uma alteração nos dados afetaria a validade da blockchain

print(f"\n{Fore.RED}{'#'*40}{Style.RESET_ALL}")
print(f"{Fore.RED}--- SIMULANDO ATAQUE ---{Style.RESET_ALL}")
print(f"{Fore.RED}{'#'*40}{Style.RESET_ALL}\n")

# Alterando os dados do Bloco 1 diretamente.
# Isso simula uma tentativa maliciosa de modificar uma transação.
original_data = my_chain.chain[1].data # Guarda os dados originais para a mensagem
my_chain.chain[1].data = {"from": "Alice", "to": "Eve", "amount": 500} # Modifica os dados
print(f"{Fore.RED}Dados do Bloco 1 alterados de '{original_data}' para '{my_chain.chain[1].data}'{Style.RESET_ALL}")

# Recalculamos o hash do bloco alterado manualmente.
# Se não fizéssemos isso, o `is_chain_valid` detectaria um hash inválido no próprio Bloco 1.
# Ao recalculá-lo, o Bloco 1 parecerá "válido" em si, mas seu novo hash não corresponderá
# ao `previous_hash` do Bloco 2, quebrando o encadeamento.
my_chain.chain[1].hash = my_chain.chain[1].calculate_hash()
print(f"{Fore.RED}Hash do Bloco 1 recalculado para '{my_chain.chain[1].hash}'{Style.RESET_ALL}\n")

# Imprimir a blockchain após a simulação de ataque (com o hash do bloco 1 recalculado)
my_chain.print_chain()

# Verificar a integridade da blockchain novamente.
# Agora, ela deve reportar que a cadeia é inválida devido à quebra de encadeamento.
my_chain.is_chain_valid()