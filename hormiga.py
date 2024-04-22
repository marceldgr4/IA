import random
import pandas as pd
import time
import matplotlib.pyplot as plt

# Lista para almacenar los resultados
df = []

# Función principal del algoritmo de optimización de colonia de hormigas (ACO)
def ant_colony_optimization(max_iterations, num_ants, pheromone_constant, evaporation_rate, capacity, weights, values, verbose=True, iter=0):
    # Inicialización de niveles de feromona
    num_items = len(weights)
    pheromones = [pheromone_constant] * num_items
    best_solution = None
    best_value = 0
    best_iter = 0

    # Listas para el gráfico de convergencia
    convergence_x = []
    convergence_y = []

    # Valor previo de la mejor solución
    prev_best_value = 0

    # Tiempo de inicio
    start = time.time()

    # Bucle principal del algoritmo
    for i in range(max_iterations):
        # Inicialización de hormigas
        ant_solutions = []
        for ant in range(num_ants):
            ant_solution = [0] * num_items
            ant_weight = 0
            ant_value = 0

            # Construir la solución de la hormiga
            while ant_weight < capacity:
                item = roulette_wheel_selection(pheromones, ant_solution, weights, capacity - ant_weight)
                if item is None:
                    break
                ant_solution[item] += 1
                ant_weight += weights[item]
                ant_value += values[item]

            ant_solutions.append((ant_solution, ant_value))

            # Actualizar la mejor solución global
            if ant_value > best_value:
                best_solution = ant_solution
                best_value = ant_value
                best_iter = i

        # Actualizar niveles de feromona
        for j in range(num_items):
            for ant_solution, ant_value in ant_solutions:
                if ant_solution[j] == 1:
                    pheromones[j] += ant_value / capacity
            pheromones[j] *= evaporation_rate

        # Registrar la mejor solución en el gráfico de convergencia
        convergence_x.append(i)
        convergence_y.append(best_value)

        # Verificar si se ha encontrado una nueva mejor solución
        if best_value != prev_best_value:
            print(f"Iteración {i}: Nuevo mejor valor = {best_value}")
            prev_best_value = best_value

        # Mostrar progreso
        if verbose and i % (max_iterations // 10) == 0:
            print(f"Iteración {i}: Mejor valor = {best_value}")

    # Graficar el gráfico de convergencia
    plt.plot(convergence_x, convergence_y)
    aux = best_value
    timer = time.time() - start
    df.append((aux, timer, best_iter))
    plt.title(f"Convergencia {iter}")
    plt.xlabel("Iteración")
    plt.ylabel("Mejor valor")
    plt.show()

    return best_solution, best_value, timer, best_iter

# Función para la selección de la ruleta
def roulette_wheel_selection(pheromones, solution, weights, remaining_capacity):
    total_pheromone = 0
    for i in range(len(pheromones)):
        if solution[i] == 0 and weights[i] <= remaining_capacity:
            total_pheromone += pheromones[i]

    if total_pheromone == 0:
        return None

    target = random.uniform(0, total_pheromone)
    cum_pheromone = 0
    for i in range(len(pheromones)):
        if solution[i] == 0 and weights[i] <= remaining_capacity:
            cum_pheromone += pheromones[i]
            if cum_pheromone >= target:
                return i

    return None

# Parámetros del algoritmo
max_iterations = 400
num_ants = 10
pheromone_constant = 5
evaporation_rate = 0.5

# Cargar datos desde un archivo Excel
data = pd.read_excel('Mochila_capacidad_maxima_10kg.xlsx',header=None)
values = list(map(float, data.iloc[1:11, 2].tolist()))
weights = list(map(int, data.iloc[1:11, 1].tolist()))
capacity = int(data.iloc[1, 0])

# Ejecutar el algoritmo 30 veces
for i in range(30):
    best_solution, best_value, crono, best_iter = ant_colony_optimization(max_iterations, num_ants, pheromone_constant,
                                                               evaporation_rate, capacity, weights, values, iter=i + 1)
    print("Mejor solución:", best_solution)
    print("Iteraciones para converger:", best_iter)
    print("Mejor valor:", best_value)
    print(f"Tiempo total: {crono} s")

# Convertir resultados a un DataFrame y guardarlos en un archivo CSV
aux_df = pd.DataFrame(df)
now = time.strftime("%Y-%m-%d_%H-%M-%S")
aux_df.to_csv(f"resultados_{now}.csv")
