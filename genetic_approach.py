from ANN import ANN, Sigmoid, ReLU, MSE_LOSS, CE_LOSS, CONSTANT_LEARNING_RATE
from preproccessing import Data
import matplotlib.pyplot as plt
from random import randint
import numpy as np
from imblearn.over_sampling import SMOTE

# process the data into training and test data
training_set, test_set = Data.get_data(na_treatment="averaged", standard=True, proportions=[0.9, 0.1])
training_inputs, training_targets = training_set
test_inputs, test_targets = test_set

smote = True
if smote:
    sm = SMOTE(k_neighbors=5)
    training_inputs, training_targets = sm.fit_resample(training_inputs, training_targets)


# parameters for this genetic approach...
pop_size = 100
max_size_of_layer = 50
max_num_hidden_layers = 2
num_to_retain = 20
allowed_activation_functions = [Sigmoid, ReLU]
allowed_learning_rates = [CONSTANT_LEARNING_RATE(0.1), CONSTANT_LEARNING_RATE(0.01), CONSTANT_LEARNING_RATE(0.001)]
allowed_loss_functions = [CE_LOSS, MSE_LOSS]
NUM_GENERATIONS = 5
EPOCHS = 100

GENERATION=0

def fitness_function(ann):
    cm = ann.confusion_matrix(test_inputs, test_targets)
    acc = (cm["tp"] + cm["tn"]) / (cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"])
    return acc

# produce an ANN which resembles the mother and father's characteristics
def offspring(mother, father, member):
    # decide whether to use the dimensions of the mom or dad
    if randint(0, 1) == 1:
        dims = mother.dims
        funs = mother.funs
    else:
        dims = father.dims
        funs = father.funs        

    # decide on loss function
    if randint(0, 1) == 1:
        loss_function = mother.loss_function
    else:
        loss_function = father.loss_function
    
    # decide on learning rate
    if randint(0, 1) == 1:
        learning_rate = mother.learning_rate_function
    else:
        learning_rate = father.learning_rate_function

    # create child
    return ANN(dims, funs, loss_function, learning_rate, name="GEN%sMEM%s" % (GENERATION, member))

# randomly change individual dimensions or functions with a small probability
def mutate(ann):
    if randint(0, 25) == 1:
        print("Mutated!")
        n_hidden_dims = len(ann.dims) - 2
        mutated_prop = randint(0, 3)
        if mutated_prop  == 0:
            ann.dims[randint(1, n_hidden_dims)] = randint(0, max_size_of_layer)
        elif mutated_prop == 1:
            ann.funs[randint(1, n_hidden_dims)] = np.random.choice(allowed_activation_functions)()
        elif mutated_prop == 2:
            ann.loss_function = np.random.choice(allowed_loss_functions)()
        else:
            ann.learning_rate = np.random.choice(allowed_learning_rates)



# create a population of randomly generated neural nets (within a certain range of reasonable values...)
population = []
for i in range(pop_size):
    # generate ANN dimensions
    num_hidden_layers = randint(1, max_num_hidden_layers)
    dims = [randint(0, max_size_of_layer) for j in range(num_hidden_layers)]
    dims.insert(0, 32)
    dims.insert(len(dims), 1)
    
    # generate the activation functions based on the dimensions
    functions = [np.random.choice(allowed_activation_functions)() for j in range(num_hidden_layers)]
    functions.append(Sigmoid())

    learning_rate = np.random.choice(allowed_learning_rates)
    loss_function = np.random.choice(allowed_loss_functions)()

    population.append(ANN(dims, functions, loss_function, learning_rate, name="GEN%sMEM%s" % (GENERATION, i)))

# iterate through generations
for gen in range(NUM_GENERATIONS):
    GENERATION += 1
    print("GENERATION %s -=-=-=-=-=-=-=-=-=-=-=-=" % GENERATION)
    fitness = []
    for i in range(pop_size):
        population[i].train(EPOCHS, training_inputs=training_inputs, training_outputs=training_targets, test_inputs=test_inputs, test_outputs=test_targets, verbose = False)
        print("trained %s" % i)

    # evolve the generation
    ranked = [(fitness_function(x), x) for x in population]
    best_acc = max(ranked, key=lambda x:x[0])
    ranked = [x[1] for x in sorted(ranked, key=lambda x: x[0], reverse=True)]

    parents = ranked[:num_to_retain]
    children = []

    for child_idx in range(pop_size - num_to_retain):
        curr_parents = np.random.choice(parents, 2, replace=False)
        children.append(offspring(curr_parents[0], curr_parents[1], child_idx))

    parents.extend(children)
    population = parents
    for net in population:
        mutate(net)

    print("Best Accuracy: %s" % str(best_acc))
    
# save the best neural net
ranked = [(fitness_function(x), x) for x in population]
ranked = [x[1] for x in sorted(ranked, key=lambda x: x[0], reverse=True)]
[np.save("./winning_weights_%s" % i, ranked[0].weights[i]) for i in range(len(ranked[0].weights))]
[np.save("./winning_biases_%s" % i, ranked[0].biases[i]) for i in range(len(ranked[0].biases))]

cm = ranked[0].confusion_matrix(test_inputs, test_targets)
print(cm)
acc = (cm["tp"] + cm["tn"]) / (cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"])
print("accuracy: %s" % acc)

ranked[0].ROC_curve(test_inputs, test_targets)
    
