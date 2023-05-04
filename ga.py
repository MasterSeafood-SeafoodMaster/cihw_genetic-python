import numpy as np
from rbfn import computeScore

class GeneticAlgorithm:
    def __init__(self, population_size, num_genes, initpop, fitness_func, mutation_rate=0.0001, crossover_rate=0.5):
        self.population_size = population_size
        self.num_genes = num_genes
        self.fitness_func = fitness_func
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = initpop

    def selection(self, fitness_scores):
        fitness_scores = np.array(fitness_scores).astype(float)
        fitness_scores /= np.sum(fitness_scores).astype(float)
        indices = np.random.choice(range(self.population_size), size=self.population_size, replace=True, p=fitness_scores)
        return self.population[indices]

    def crossover(self, parents):
        children = np.zeros((self.population_size, self.num_genes))
        for i in range(0, self.population_size, 2):
            if np.random.uniform(0, 1) < self.crossover_rate:
                crossover_point = np.random.randint(1, self.num_genes-1)
                children[i] = np.concatenate((parents[i][:crossover_point], parents[i+1][crossover_point:]))
                children[i+1] = np.concatenate((parents[i+1][:crossover_point], parents[i][crossover_point:]))
            else:
                children[i] = parents[i]
                children[i+1] = parents[i+1]
        return children

    def mutation(self, population):
        for i in range(self.population_size):
            for j in range(self.num_genes):
                if np.random.uniform(0, 1) < self.mutation_rate:
                    population[i,j] = 1 - population[i,j]
        return population

    def evolve(self, i):
        fitness_scores = [self.fitness_func(individual) for individual in self.population]
        parents = self.selection(fitness_scores)
        children = self.crossover(parents)
        self.population = self.mutation(children)
        s = "evolve_"+str(i)+" current best:"
        print(s)
        self.current_max = self.population[np.argmax([fitness_func(individual) for individual in self.population])]
        print(self.current_max)

def fitness_func(individual):
    return computeScore(individual)


def getbestDNA(pop, ps, ng):
    ga = GeneticAlgorithm(population_size=ps, num_genes=ng, initpop=pop, fitness_func=fitness_func)
    for i in range(10):
        ga.evolve(i)

    best_individual = ga.population[np.argmax([fitness_func(individual) for individual in ga.population])]
    return best_individual