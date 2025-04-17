import time
import tracemalloc
import pandas as pd
from collections import deque
import heapq

# =========================
# 1. Leitura do CSV
# =========================
df = pd.read_csv("codigo/ed02-puzzle8.csv")
estados_iniciais = df.values.tolist()

# =========================
# 2. Definições Comuns
# =========================
estado_objetivo = [1, 2, 3, 4, 5, 6, 7, 8, 0]


def obter_vizinhos(estado):
    vizinhos = []
    indice_zero = estado.index(0)
    linha, coluna = divmod(indice_zero, 3)
    movimentos = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # cima, baixo, esquerda, direita
    for dr, dc in movimentos:
        r, c = linha + dr, coluna + dc
        if 0 <= r < 3 and 0 <= c < 3:
            novo_indice = r * 3 + c
            novo_estado = estado[:]
            novo_estado[indice_zero], novo_estado[novo_indice] = novo_estado[novo_indice], novo_estado[indice_zero]
            vizinhos.append(novo_estado)
    return vizinhos


def distancia_manhattan(estado):
    distancia = 0
    for i, val in enumerate(estado):
        if val != 0:
            indice_objetivo = estado_objetivo.index(val)
            x1, y1 = divmod(i, 3)
            x2, y2 = divmod(indice_objetivo, 3)
            distancia += abs(x1 - x2) + abs(y1 - y2)
    return distancia


# =========================
# 3. Algoritmos de Busca
# =========================

# Busca em Largura (BFS)
def bfs(estado_inicial):
    inicio = tuple(estado_inicial)
    objetivo = tuple(estado_objetivo)
    visitados = set()
    fila = deque([(inicio, [], 0)])
    while fila:
        atual, caminho, profundidade = fila.popleft()
        if atual == objetivo:
            return caminho + [atual]
        if atual not in visitados:
            visitados.add(atual)
            for vizinho in obter_vizinhos(list(atual)):
                fila.append((tuple(vizinho), caminho + [atual], profundidade + 1))
    return None


# Busca em Profundidade (DFS)
def dfs(estado_inicial, profundidade_maxima=50):
    inicio = tuple(estado_inicial)
    objetivo = tuple(estado_objetivo)
    visitados = set()
    pilha = [(inicio, [], 0)]
    while pilha:
        atual, caminho, profundidade = pilha.pop()
        if atual == objetivo:
            return caminho + [atual]
        if atual not in visitados and profundidade < profundidade_maxima:
            visitados.add(atual)
            for vizinho in obter_vizinhos(list(atual)):
                pilha.append((tuple(vizinho), caminho + [atual], profundidade + 1))
    return None


# Algoritmo Guloso
def gulosa(estado_inicial):
    inicio = tuple(estado_inicial)
    objetivo = tuple(estado_objetivo)
    visitados = set()
    heap = [(distancia_manhattan(estado_inicial), inicio, [])]
    while heap:
        h, atual, caminho = heapq.heappop(heap)
        if atual == objetivo:
            return caminho + [atual]
        if atual not in visitados:
            visitados.add(atual)
            for vizinho in obter_vizinhos(list(atual)):
                heapq.heappush(heap, (distancia_manhattan(vizinho), tuple(vizinho), caminho + [atual]))
    return None


# Algoritmo A*
def a_star(estado_inicial):
    inicio = tuple(estado_inicial)
    objetivo = tuple(estado_objetivo)
    visitados = set()
    heap = [(distancia_manhattan(estado_inicial), 0, inicio, [])]
    while heap:
        f, g, atual, caminho = heapq.heappop(heap)
        if atual == objetivo:
            return caminho + [atual]
        if atual not in visitados:
            visitados.add(atual)
            for vizinho in obter_vizinhos(list(atual)):
                novo_g = g + 1
                novo_f = novo_g + distancia_manhattan(vizinho)
                heapq.heappush(heap, (novo_f, novo_g, tuple(vizinho), caminho + [atual]))
    return None


# =========================
# 4. Avaliação
# =========================
def avaliar_algoritmo(funcao_algoritmo, estado):
    tracemalloc.start()
    inicio_tempo = time.time()
    caminho = funcao_algoritmo(estado)
    fim_tempo = time.time()
    atual, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "Movimentos": len(caminho) - 1 if caminho else -1,
        "Tempo (s)": round(fim_tempo - inicio_tempo, 6),
        "Memória (KB)": round(pico / 1024, 2),
    }


# =========================
# 5. Execução dos Testes
# =========================
resultados = []

for i, estado in enumerate(estados_iniciais):
    linha = {"Instância": i + 1}
    for nome, algoritmo in [
        ("BFS", bfs),
        ("DFS", dfs),
        ("Gulosa", gulosa),
        ("A*", a_star),
    ]:
        try:
            resultado = avaliar_algoritmo(algoritmo, estado)
            linha.update({
                f"{nome} Movimentos": resultado["Movimentos"],
                f"{nome} Tempo (s)": resultado["Tempo (s)"],
                f"{nome} Memória (KB)": resultado["Memória (KB)"]
            })
        except Exception as e:
            linha.update({
                f"{nome} Movimentos": "Erro",
                f"{nome} Tempo (s)": "Erro",
                f"{nome} Memória (KB)": "Erro"
            })
    resultados.append(linha)

# =========================
# 6. Exportar Resultados
# =========================
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("comparacao_algoritmos_puzzle8.csv", index=False)
print("Comparação final salva em 'comparacao_algoritmos_puzzle8.csv'.")
