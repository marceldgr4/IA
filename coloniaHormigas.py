import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AntColony:
    def __init__(self, items, max_weight, n_ants, evaporation_rate=0.5, alpha=1, beta=1, iterations=100):
        self.items = items
        self.max_weight = max_weight
        self.n_ants = n_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.best_solution = None
        self.best_value = float('-inf')
        self.convergence = []

    def select_items(self, pheromones):
        selected_items = []
        total_value = 0
        total_weight = 0

        while total_weight <= self.max_weight:
            probs = [((pheromones[i] ** self.alpha) * ((self.items.loc[i, 'Valor'] / self.items.loc[i, 'Peso_kg']) ** self.beta)) for i in range(len(self.items))]
            total_prob = sum(probs)
            cum_prob = np.cumsum(probs) / total_prob
            r = random.random()

            selected_item = None
            for i, prob in enumerate(cum_prob):
                if r <= prob:
                    selected_item = i
                    break

            if selected_item is None:
                break

            selected_items.append(selected_item)
            total_value += self.items.loc[selected_item, 'Valor']
            total_weight += self.items.loc[selected_item, 'Peso_kg']

        return selected_items[:-1], total_value - self.items.loc[selected_items[-1], 'Valor'], total_weight - self.items.loc[selected_items[-1], 'Peso_kg']

    def evaporate_pheromones(self, pheromones):
        return [(1 - self.evaporation_rate) * p for p in pheromones]

    def update_pheromones(self, pheromones, selected_items, value):
        for item in selected_items:
            pheromones[item] += value / self.items.loc[item, 'Valor']
        return pheromones

    def run(self):
        pheromones = [1] * len(self.items)

        for _ in range(self.iterations):
            solutions = []
            for _ in range(self.n_ants):
                selected_items, value, weight = self.select_items(pheromones)
                solutions.append((selected_items, value))

                if weight <= self.max_weight and value > self.best_value:
                    self.best_solution = selected_items
                    self.best_value = value

            self.convergence.append(self.best_value)

            pheromones = self.evaporate_pheromones(pheromones)
            for selected_items, value in solutions:
                pheromones = self.update_pheromones(pheromones, selected_items, value)

        return self.best_solution, self.best_value

def plot_convergence(convergence):
    plt.figure(figsize=(10, 6))
    plt.plot(convergence)
    plt.title('Convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Best Value')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Leer datos desde archivo Excel
    df = pd.read_excel('Mochila_capacidad_maxima_10kg.xlsx', engine='openpyxl')
    items = df[['Peso_kg', 'Valor', 'Cantidad']].values.tolist()

    # Expandir items por cantidad
    expanded_items = []
    for i, (peso, valor, cantidad) in enumerate(items):
        expanded_items.extend([(peso, valor)] * int (cantidad))

    items_df = pd.DataFrame(expanded_items, columns=['Peso_kg', 'Valor'])

    max_weight = 10.0
    n_ants = 2
    iterations = 200

    ant_colony = AntColony(items_df, max_weight, n_ants, iterations=iterations)
    best_solution, best_value = ant_colony.run()

    print(f"Best solution: {best_solution}")
    print(f"Best value: {best_value}")
    print(f"Number of iterations: {iterations}")
    plot_convergence(ant_colony.convergence)
