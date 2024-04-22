import random
import matplotlib.pyplot as plt
import pandas as pd
import time



# Leer archivo xlsx
data = pd.read_excel('Mochila_capacidad_maxima_10kg.xlsx')
values = list(map(int, data.iloc[1:11, 1].tolist()))
weights = list(map(int, data.iloc[1:11, 2].tolist()))
max_weight = int(data.iloc[1, 0])


# Genetic algorithm parameters
population_size = 100
mutation_rate = 0.05
crossover_rate = 0.5
elite_size =10
generations = 300
convergence = []
# Generate initial population
def generate_population(size):
    population = []
    for i in range(size):
        chromosome = [random.randint(0, 10) for _ in range(len(weights))]
        population.append(chromosome)
    return population

# Calculate fitness of each chromosome
def fitness(chromosome):
    total_weight = sum([weights[i] * chromosome[i] for i in range(len(weights))])
    if total_weight > max_weight:
        return 0
    else:
        return sum([values[i] * chromosome[i] for i in range(len(values))])

# Perform single-point crossover between two parents
def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    else:
        return parent1, parent2

# Perform mutation on a chromosome
def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]

# Select a chromosome from the population using tournament selection
def tournament_selection(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    best_chromosome = None
    best_fitness = -1
    for chromosome in tournament:
        chromosome_fitness = fitness(chromosome)
        if chromosome_fitness > best_fitness:
            best_chromosome = chromosome
            best_fitness = chromosome_fitness
    return best_chromosome

# Perform genetic algorithm
def genetic_algorithm(population_size, mutation_rate, crossover_rate, elite_size, generations):
    population = generate_population(population_size)
    # time
    start = time.time()
    is_converged = False
    max_converged = 0
    for i in range(generations):
        # Select parents for reproduction
        parent1 = tournament_selection(population, 2)
        parent2 = tournament_selection(population, 2)
        # Perform crossover and mutation to create new offspring
        child1, child2 = crossover(parent1, parent2)
        mutate(child1)
        mutate(child2)
        # Replace weakest individuals in population with offspring
        fitnesses = [fitness(chromosome) for chromosome in population]
        min_fitness_index = fitnesses.index(min(fitnesses))
        if fitness(child1) > fitnesses[min_fitness_index]:
            population[min_fitness_index] = child1
        min_fitness_index = fitnesses.index(min(fitnesses))
        if fitness(child2) > fitnesses[min_fitness_index]:
            population[min_fitness_index] = child2
        # Select elite individuals to survive to next generation
        elite_population = sorted(population, key=fitness, reverse=True)[:elite_size]
        # Calculate convergence and print progress
        convergence.append(max(fitnesses))
       
        print("Generation", i + 1, "- Best fitness:", max(fitnesses))
        if not is_converged and len(set(convergence)) == 1:
            # print("The algorithm converged at generation", i+1)
            max_converged = i+1
            is_converged = True
        else:
            is_converged = False
    return max(fitnesses), time.time() - start, max_converged



df = []
convergence = []
convergence_iters = []

for i in range(30):
    best_fitness, crono, converged_iter = genetic_algorithm(population_size, mutation_rate, crossover_rate, elite_size, generations)
    print(f"Best fitness: {best_fitness}")
    print(f"Total time: {crono} s")
    print(f"Convergence iteration: {converged_iter}")
    plt.plot(convergence)
    plt.title(f"Iteración No° {i + 1}")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()

    if converged_iter is not None:
        if best_fitness > 0:
            df.append([i+1, best_fitness, crono, converged_iter if converged_iter is not None else 0])

        else:
            df.append([i+1, 0, crono, converged_iter])
            convergence_iters.append((i+1, converged_iter))


    convergence = []


resultados_df = pd.DataFrame(df, columns=['Iteration', 'Best Fitness', 'Time (s)', 'Convergence Iteration'])
now = time.strftime("%Y-%m-%d_%H-%M-%S")
resultados_df.to_csv(f"resultados_{now}.csv", index=False)