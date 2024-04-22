

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# archivo
data = pd.read_excel('Mochila_capacidad_maxima_10kg.xlsx', header=None)
values = list(map(int, data.iloc[1:11, 2].tolist()))
weights = list(map(float, data.iloc[1:11, 1].tolist()))
capacity = 10
n_items = len(weights)


# Define the simulated annealing function
def simulated_annealing(weights, values, capacity, n_items, t_init=30, t_min=0.01, alpha=0.98, n_iter=1000):
    # Initialize the current state
    cur_state = np.zeros(n_items, dtype=int)

    # Define the objective function
    def objective(state):
        # Calculate the total weight and value of the state
        total_weight = np.sum(weights * state)
        total_value = np.sum(values * state)

        # Return 0 if the state violates the capacity constraint
        if total_weight > capacity:
            return 0

        # Return the total value of the state
        return total_value

    # Initialize the best state
    best_state = cur_state.copy()
    best_obj = objective(cur_state)
    obj_vals = [best_obj]

    # Initialize the temperature
    t = t_init

    # Iterate until convergence
    start = time.time()
    iter_convergence = -1
    for i in range(n_iter):
        # Generate a new state by randomly flipping one item
        new_state = cur_state.copy()
        idx = np.random.randint(n_items)
        new_state[idx] = 1 - new_state[idx]

        # Calculate the objective function of the new state
        new_obj = objective(new_state)

        # Calculate the acceptance probability
        delta = new_obj - best_obj
        if delta > 0:
            accept_prob = 1
        else:
            accept_prob = np.exp(delta / t)

        # Accept or reject the new state
        if np.random.rand() < accept_prob:
            cur_state = new_state
            cur_obj = new_obj
            if new_obj > best_obj:
                best_state = new_state
                best_obj = new_obj
                obj_vals.append(best_obj)
        else:
            obj_vals.append(best_obj)

        # Update the temperature
        t *= alpha
        if t < t_min:
            break
        elif iter_convergence == -1 and best_obj > 0:
            if np.abs(obj_vals[-2] - best_obj) < 1e-9:
                iter_convergence = i

        print

    # Return the best state and its objective value
    return best_state, best_obj, obj_vals, time.time() - start, iter_convergence


df = []
# Run the simulated annealing algorithm
for i in range(30):
    best_state, best_obj, obj_vals, crono, iter_convergence = simulated_annealing(weights, values, capacity, n_items, t_init=100, t_min=0.01, alpha=0.99, n_iter=1000)

    # Print the results
    print(f"Best state {i+1}:", best_state)
    print(f"Best objective value {i+1}:", best_obj)
    print(f"Total time {i+1}:", crono, "s")
    print(f"Iterations to converge:", iter_convergence)

    # Plot the convergence graph
    plt.plot(obj_vals)
    plt.xlabel('Iterations')
    plt.ylabel
    plt.title(f'Intento N°{i+1}')
    plt.show()
    df.append([i+1, best_obj, crono, iter_convergence])
    

resultados_df = pd.DataFrame(df, columns=['Ejecucion', 'Best objective value', 'Time (s)', 'Convergence Iteration'])
now = time.strftime("%Y-%m-%d_%H-%M-%S")
resultados_df.to_csv(f"resultados_{now}.csv", index=False)
