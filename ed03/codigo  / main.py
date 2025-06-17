import pandas as pd
import numpy as np
import random
import time
import os
from typing import List

# ===================== Função para ler o CSV =====================
def load_function(csv_path):
    df = pd.read_csv(csv_path)
    coef = df['Coeficiente'].values
    return coef

# ===================== Função objetivo =====================
def evaluate(individual: List[float], coefficients: List[float]):
    return sum(x * c for x, c in zip(individual, coefficients))

# ===================== Operadores Genéticos =====================
def one_point_crossover(p1, p2):
    point = random.randint(1, len(p1) - 1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

def two_point_crossover(p1, p2):
    p1_idx, p2_idx = sorted(random.sample(range(len(p1)), 2))
    return (p1[:p1_idx] + p2[p1_idx:p2_idx] + p1[p2_idx:],
            p2[:p1_idx] + p1[p1_idx:p2_idx] + p2[p2_idx:])

def uniform_crossover(p1, p2):
    child1, child2 = [], []
    for a, b in zip(p1, p2):
        if random.random() < 0.5:
            child1.append(a)
            child2.append(b)
        else:
            child1.append(b)
            child2.append(a)
    return child1, child2

def mutate(individual, mutation_rate, bounds):
    return [x + random.uniform(-1, 1) if random.random() < mutation_rate else x for x in individual]

# ===================== Inicialização =====================
def random_population(size, num_vars, bounds):
    return [[random.uniform(*bounds) for _ in range(num_vars)] for _ in range(size)]

# ===================== Critério de parada =====================
def has_converged(history, epsilon=1e-4, window=10):
    if len(history) < window:
        return False
    recent = history[-window:]
    return max(recent) - min(recent) < epsilon

# ===================== AG Principal =====================
def genetic_algorithm(coefficients, objective='max',
                      crossover_type='ponto_um', mutation_rate=0.01,
                      population_size=50, generations=100,
                      bounds=(-10, 10)):
    num_vars = len(coefficients)
    population = random_population(population_size, num_vars, bounds)
    crossover_func = {
        'ponto_um': one_point_crossover,
        'dois_pontos': two_point_crossover,
        'uniforme': uniform_crossover
    }[crossover_type]

    best_fitness_history = []
    best_individual = None
    best_fitness = float('-inf') if objective == 'max' else float('inf')

    for gen in range(generations):
        fitness = [evaluate(ind, coefficients) for ind in population]
        if objective == 'min':
            fitness = [-f for f in fitness]

        # Seleção por torneio binário
        selected = [max(random.sample(list(zip(population, fitness)), 2), key=lambda x: x[1])[0] for _ in population]

        # Crossover e mutação
        new_population = []
        for i in range(0, len(selected), 2):
            p1, p2 = selected[i], selected[(i+1) % len(selected)]
            c1, c2 = crossover_func(p1, p2)
            new_population.extend([mutate(c1, mutation_rate, bounds), mutate(c2, mutation_rate, bounds)])

        population = new_population[:population_size]

        current_best = max(fitness) if objective == 'max' else -min(fitness)
        current_individual = population[np.argmax(fitness)] if objective == 'max' else population[np.argmin(fitness)]

        best_fitness_history.append(current_best)
        if (objective == 'max' and current_best > best_fitness) or (objective == 'min' and current_best < best_fitness):
            best_fitness = current_best
            best_individual = current_individual

        if has_converged(best_fitness_history):
            break

    return best_individual, best_fitness, gen + 1

# ===================== Execução em lote =====================
def run_all(csv_folder='codigo/csvs', output_csv='resultados.csv'): 
    results = []
    files = [f for f in os.listdir(csv_folder) if f.startswith('function_opt') and f.endswith('.csv')]

    for file in files:
        coef = load_function(os.path.join(csv_folder, file))
        for objective in ['min', 'max']:
            for crossover in ['ponto_um', 'dois_pontos', 'uniforme']:
                for mutation in [0.01, 0.05, 0.1]:
                    start = time.time()
                    ind, val, gens = genetic_algorithm(coef, objective=objective,
                                                       crossover_type=crossover,
                                                       mutation_rate=mutation)
                    end = time.time()
                    results.append({
                        'arquivo': file,
                        'objetivo': objective,
                        'crossover': crossover,
                        'mutacao': mutation,
                        'valor': val,
                        'geracoes': gens,
                        'tempo': round(end - start, 4)
                    })
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Arquivo '{output_csv}' foi gerado com sucesso!")

if __name__ == '__main__':
    run_all()
    
