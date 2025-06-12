import random
import time
import pandas as pd

# ----------------------------- Configuração do Problema -----------------------------
ITEMS = [(10, 60), (20, 100), (30, 120), (5, 30), (15, 70)]
CAPACIDADE_MAXIMA = 50
POPULACAO_TAMANHO = 30
GERACOES = 100

# ----------------------------- Funções de Apoio -----------------------------
def fitness(individuo):
    peso_total, valor_total = 0, 0
    for gene, (peso, valor) in zip(individuo, ITEMS):
        if gene == 1:
            peso_total += peso
            valor_total += valor
    return valor_total if peso_total <= CAPACIDADE_MAXIMA else 0

def gerar_individuo_aleatorio():
    return [random.randint(0, 1) for _ in range(len(ITEMS))]

def gerar_individuo_heuristico():
    ratio = sorted([(i, v/p) for i, (p, v) in enumerate(ITEMS)], key=lambda x: -x[1])
    individuo = [0] * len(ITEMS)
    peso_total = 0
    for i, _ in ratio:
        if peso_total + ITEMS[i][0] <= CAPACIDADE_MAXIMA:
            individuo[i] = 1
            peso_total += ITEMS[i][0]
    return individuo

def inicializar_populacao(metodo='aleatorio'):
    if metodo == 'aleatorio':
        return [gerar_individuo_aleatorio() for _ in range(POPULACAO_TAMANHO)]
    else:
        return [gerar_individuo_heuristico() for _ in range(POPULACAO_TAMANHO)]

def selecao_torneio(populacao, k=3):
    selecionados = random.sample(populacao, k)
    return max(selecionados, key=fitness)

def crossover(pai1, pai2, tipo='um_ponto'):
    if tipo == 'um_ponto':
        ponto = random.randint(1, len(pai1) - 1)
        return pai1[:ponto] + pai2[ponto:], pai2[:ponto] + pai1[ponto:]
    elif tipo == 'dois_pontos':
        p1, p2 = sorted(random.sample(range(len(pai1)), 2))
        filho1 = pai1[:p1] + pai2[p1:p2] + pai1[p2:]
        filho2 = pai2[:p1] + pai1[p1:p2] + pai2[p2:]
        return filho1, filho2
    elif tipo == 'uniforme':
        filho1, filho2 = [], []
        for g1, g2 in zip(pai1, pai2):
            if random.random() < 0.5:
                filho1.append(g1)
                filho2.append(g2)
            else:
                filho1.append(g2)
                filho2.append(g1)
        return filho1, filho2

def mutacao(individuo, taxa):
    return [gene if random.random() > taxa else 1 - gene for gene in individuo]

# ----------------------------- Algoritmo Genético -----------------------------
def algoritmo_genetico(config):
    populacao = inicializar_populacao(config['inicializacao'])
    melhor_individuo = max(populacao, key=fitness)
    historico = [fitness(melhor_individuo)]

    for _ in range(GERACOES):
        nova_populacao = []
        while len(nova_populacao) < POPULACAO_TAMANHO:
            pai1 = selecao_torneio(populacao)
            pai2 = selecao_torneio(populacao)
            filho1, filho2 = crossover(pai1, pai2, config['crossover'])
            filho1 = mutacao(filho1, config['mutacao'])
            filho2 = mutacao(filho2, config['mutacao'])
            nova_populacao.extend([filho1, filho2])
        populacao = nova_populacao[:POPULACAO_TAMANHO]
        atual_melhor = max(populacao, key=fitness)
        historico.append(fitness(atual_melhor))
        if config['parada'] == 'convergencia' and len(historico) > 5 and len(set(historico[-5:])) == 1:
            break

    return fitness(max(populacao, key=fitness)), historico[-1]

# ----------------------------- Execução e Comparação -----------------------------
configuracoes = []
tipos_crossover = ['um_ponto', 'dois_pontos', 'uniforme']
taxas_mutacao = [0.01, 0.1, 0.3]
inicializacoes = ['aleatorio', 'heuristica']
criterios_parada = ['geracoes', 'convergencia']

for c in tipos_crossover:
    for m in taxas_mutacao:
        for i in inicializacoes:
            for p in criterios_parada:
                configuracoes.append({
                    'crossover': c,
                    'mutacao': m,
                    'inicializacao': i,
                    'parada': p
                })

resultados = []
for cfg in configuracoes:
    inicio = time.time()
    valor_final, _ = algoritmo_genetico(cfg)
    fim = time.time()
    resultados.append({
        'Crossover': cfg['crossover'],
        'Mutação': cfg['mutacao'],
        'Inicialização': cfg['inicializacao'],
        'Critério Parada': cfg['parada'],
        'Valor Final': valor_final,
        'Tempo (s)': round(fim - inicio, 4)
    })

df_resultados = pd.DataFrame(resultados)
df_resultados.sort_values(by='Valor Final', ascending=False, inplace=True)
df_resultados.reset_index(drop=True, inplace=True)

# ----------------------------- Exportar CSV -----------------------------
df_resultados.to_csv("resultados.csv", index=False)
print("Arquivo 'resultados.csv' gerado com sucesso!")
print(df_resultados.head(10))

