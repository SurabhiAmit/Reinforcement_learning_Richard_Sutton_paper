import numpy as np
import random
import collections
import matplotlib.pyplot as plt


def get_arg(list, element):
    for i in range(len(list)-1):
        if list[i] == element:
            return i
    return None


def get_x():
    #random.seed(seed) #can change seeds if needed
    return random.choice([1, -1])


def random_seq():
    total_walk = ['A','B','C','D','E','F','G']
    sequence = ['D']
    while sequence[-1] != 'A' and sequence[-1] != 'G':
        k = get_arg(total_walk, sequence[-1])
        if k:
            x = get_x()
            sequence.extend(total_walk[k + x])
    return sequence


def training_set():
    count = 10
    set = []
    for i in range(count):
        set.append(random_seq())
    return set


def training_sets():
    count = 100
    sets = []
    for i in range(count):
        sets.append(training_set())
    return sets


def get_P(seq_element,w):
    xt=[0,0,0,0,0]
    sum=0
    if seq_element=='B':
        xt= [1,0,0,0,0]
    elif seq_element=='C':
        xt= [0,1,0,0,0]
    elif seq_element=='D':
        xt= [0,0,1,0,0]
    elif seq_element=='E':
        xt= [0,0,0,1,0]
    elif seq_element=='F':
        xt= [0,0,0,0,1]
    elif seq_element=='A':
        return 0,xt
    elif seq_element=='G':
        return 1,xt
    for i in range(len(w)):
        sum += xt[i]*w[i]  #w_transpose*xt
    return sum, xt


def get_w_from_seq(seq,w,lambda1):
    e_prev, et, delta_W = [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]
    for i in range(len(seq)-1):
        pt, xt = get_P(seq[i], w)
        pt_1, xt_1 = get_P(seq[i+1], w)
        p_delta = pt_1 - pt
        #et = lambda*e_prev+xt
        for k in range(len(e_prev)):
            et[k] = e_prev[k]*lambda1+xt[k]
            delta_W[k] += p_delta*et[k]
        e_prev = list(et)
    return delta_W


def update_from_set(set,w,lambda1):
    #add delta_w from each sequence component-wise
    delta_w=[0,0,0,0,0]
    for each in set:
        w_seq = get_w_from_seq(each,w,lambda1)
        for i in range(len(w_seq)):
            delta_w[i] += w_seq[i]
    return [element/10.0 for element in delta_w]


def calc_RMSE(predicted, ideal):
    i, total =0, 0
    for i in range(len(predicted)):
        error = predicted[i]-ideal[i]
        total += (error*error)
    total_m = total/len(predicted)
    return np.sqrt(total_m)


#EXPERIMENT 1
def random_walk(lambda1, data):
    ideal = [1.0/6, 1.0/3, 1.0/2, 2.0/3, 5.0/6] #probability for right-side termination
    alpha = 0.08
    min_error = 0.0001
    w =[0.5, 0.5, 0.5, 0.5, 0.5] # number of non-terminal states
    #w = [0, 0, 0, 0, 0] #checked another initialization of the weight vectors
    sets = data
    n = len(sets)
    i = 0
    converges = False
    total_rmse = []
    while i < n:
        #get w for each set
        while not converges:
            prev_w = list(w)
            delta_w = update_from_set(sets[i], w, lambda1)
            k = 0
            while k < len(w):
                w[k] += alpha*delta_w[k]
                k += 1
            error = calc_RMSE(prev_w,w)
            if error < min_error:
                converges = True
                break
        total_rmse.append(calc_RMSE(w, ideal))
        i += 1
        converges = False
    average_error = sum(total_rmse)/n
    return average_error


#EXPERIMENT 2
def experiment2(lambda2,data2,alpha):
    ideal = [1.0 / 6, 1.0 / 3, 1.0 / 2, 2.0 / 3, 5.0 / 6]  # probability for right-side termination
    w = [0.5, 0.5, 0.5, 0.5, 0.5]  # number of non-terminal states
    sets = data2
    n = len(sets)
    i = 0
    total_rmse = []
    while i < n:
        w = [0.5, 0.5, 0.5, 0.5, 0.5]
        for j in range(len(sets[i])):
            prev_w = list(w)
            delta_w = get_w_from_seq(sets[i][j], w, lambda2)
            k = 0
            while k < len(w):
                w[k] += alpha * delta_w[k]
                k += 1
        total_rmse.append(calc_RMSE(w, ideal))
        i += 1
    average_error = sum(total_rmse) / n
    return average_error


#EXPERIMENT 1
lambdas_exp1 = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

RMSE = {}
data = training_sets()
for i in range(len(lambdas_exp1)):
    RMSE[lambdas_exp1[i]] = (random_walk(lambdas_exp1[i], data))
RMS = collections.OrderedDict(sorted(RMSE.items()))

RMSES = sorted(RMSE.items())
x,y = zip(*RMSES)
plt.plot(x, y, marker='o')
plt.xlabel('lambda')
plt.ylabel('ERROR')
plt.title('Variation of RMS error with lambda for alpha=0.08')
plt.legend()
plt.grid()
#plt.text(0.56,0.24,'Widrow-Hoff (lambda=1)')
plt.show()


#EXPERIMENT 2
lambdas_exp2 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]
alphas = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]

RMSE1 = {}
data = training_sets()
for i in range(len(alphas)):
    RMSE1[alphas[i]] = (experiment2(1, data, alphas[i]))

RMSES1 = sorted(RMSE1.items())
x, y = zip(*RMSES1)
plt.plot(x, y, label="lambda = 1 (Widrow-Hoff)", marker='o')

RMSE2 = {}
for i in range(len(alphas)):
    RMSE2[alphas[i]] = (experiment2(0.8, data, alphas[i]))

RMSES2 = sorted(RMSE2.items())
x,y = zip(*RMSES2)
plt.plot(x, y, label="lambda = 0.8", marker='o')


RMSE3 = {}
for i in range(len(alphas)):
    RMSE3[alphas[i]] = (experiment2(0.3, data, alphas[i]))

RMSES3 = sorted(RMSE3.items())
x, y = zip(*RMSES3)
plt.plot(x, y, label="lambda = 0.3", marker='o')

RMSE4 = {}
for i in range(len(alphas)):
    RMSE4[alphas[i]] = (experiment2(0, data, alphas[i]))

RMSES4 = sorted(RMSE4.items())
x, y = zip(*RMSES4)
plt.plot(x, y, label="lambda = 0", marker='o')

plt.xlabel('alpha')
plt.ylabel('ERROR')
plt.title('Experiment 2: Variation of RMS error with alpha')
plt.legend()
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, 0, 0.7))
plt.show()

#EXPERIMENT 2
def least_value(dict):
    least_key = min(dict, key=dict.get)
    return dict[least_key]
RMSE = {}
for i in range(len(lambdas_exp2)):
    RMSE[lambdas_exp2[i]] = {}
    for j in range(len(alphas)):
        RMSE[lambdas_exp2[i]][alphas[j]] = (experiment2(lambdas_exp2[i], data, alphas[j]))

least_error = {}
for each in RMSE.keys():
    least_error[each] = least_value(RMSE[each])
RMSES = sorted(least_error.items())
x, y = zip(*RMSES)
plt.plot(x, y, marker='o')
plt.xlabel('lambda')
plt.ylabel('Error using best alpha')
plt.grid()
#plt.text(0.56,0.24,'Widrow-Hoff (lambda = 1)')
plt.title('Experiment 2: Variation of RMS error with lambda')
plt.legend()
plt.show()


