import random
import math
# import numpy as np
import flowersdata as data
input_count, hidden_count, output_count = 2,4,3
learning_rate = 0.4
epochs = 5000
# def init_params(input_count, hidden_count, output_count):
w_i_h = [[random.random() - 0.5 for _ in range(input_count)] for _ in range(hidden_count)]
w_h_o = [[random.random() - 0.5 for _ in range(hidden_count)] for _ in range(output_count)]
b_i_h = [0 for _ in range(hidden_count)]
b_h_o = [0 for _ in range(output_count)]
# return w_i_h, w_h_o, b_i_h, b_h_o


# def ReLU(Z):
#     return np.maximum(Z, 0)

def log_loss(activations,targets):
    losses = [-t * math.log(a) - (1 - t) * math.log(1-a) for a,t in zip(activations,targets)]
    return sum(losses)

def softmax(predictions):
    m = max(predictions)
    temp = [math.exp(p - m) for p in predictions]
    total = sum(temp)
    return [t / total for t in temp]


def forward_prop(w_i_h, b_i_h, w_h_o, b_h_o, inputs):
    pred_h = [[sum([w * a for w, a in zip(weights, inp)]) + bias for weights, bias in zip(w_i_h, b_i_h)] for inp in
              inputs]
    act_h = [[max(0, p) for p in pred] for pred in pred_h]
    pred_o = [[sum([w * a for w, a in zip(weights, inp)]) + bias for weights, bias in zip(w_h_o, b_h_o)] for inp in
              act_h]
    act_o = [softmax(predictions) for predictions in pred_o]

    return pred_h, act_h, pred_o, act_o


def ReLU_deriv(Z):
    return Z > 0


# def one_hot(Y):
#     one_hot_Y = np.zeros((Y.size, Y.max() + 1))
#     one_hot_Y[np.arange(Y.size), Y] = 1
#     one_hot_Y = one_hot_Y.T
#     return one_hot_Y


def backward_prop(pred_h, act_h, act_o, w_h_o, inputs, targets):
    errors_d_o = [[a - t for a, t in zip(ac, ta)] for ac, ta in zip(act_o, targets)]
    w_h_o_T = list(zip(*w_h_o))
    errors_d_h = [
        [sum([d * w for d, w in zip(deltas, weights)]) * (0 if p <= 0 else 1) for weights, p in zip(w_h_o_T, pred)] for
        deltas, pred in zip(errors_d_o, pred_h)]
    # print(errors_d_o)
    # gradient hidden-> output
    act_h_T = list(zip(*act_h))
    errors_d_o_T = list(zip(*errors_d_o))
    w_h_o_d = [[sum([d * a for d, a in zip(deltas, act)]) for deltas in errors_d_o_T] for act in act_h_T]
    b_h_o_d = [sum([d for d in deltas]) for deltas in errors_d_o_T]
    # print(b_h_o_d)

    # Gradient input ->hidden
    inputs_T = list(zip(*inputs))
    errors_d_h_T = list(zip(*errors_d_h))
    w_i_h_d = [[sum([d * a for d, a in zip(deltas, act)]) for deltas in errors_d_h_T] for act in inputs_T]
    b_i_h_d = [sum([d for d in deltas]) for deltas in errors_d_h_T]
    return w_h_o_d, b_h_o_d, w_i_h_d, b_i_h_d


def update_params(w_i_h, b_i_h, w_h_o, b_h_o, w_i_h_d, b_i_h_d, w_h_o_d, b_h_o_d, learning_rate, input_count,
                  hidden_count, output_count, inputs):
    w_h_o_d_T = list(zip(*w_h_o_d))
    for y in range(output_count):
        for x in range(hidden_count):
            w_h_o[y][x] -= learning_rate * w_h_o_d_T[y][x] / len(inputs)
        b_h_o[y] -= learning_rate * b_h_o_d[y] / len(inputs)

    w_i_h_d_T = list(zip(*w_i_h_d))
    for y in range(hidden_count):
        for x in range(input_count):
            w_i_h[y][x] -= learning_rate * w_i_h_d_T[y][x] / len(inputs)
        b_i_h[y] -= learning_rate * b_i_h_d[y] / len(inputs)
    return w_i_h, b_i_h, w_h_o, b_h_o


# def get_predictions(A2):
#     return np.argmax(A2, 0)
#
#
# def get_accuracy(predictions, Y):
#     print(predictions, Y)
#     return np.sum(predictions == Y) / Y.size


def gradient_descent(w_i_h,b_i_h,w_h_o,b_h_o,inputs, targets, learning_rate, epochs):
    # w_i_h, b_i_h, w_h_o, b_h_o = init_params(2, 4, 3)
    for i in range(epochs):
        pred_h, act_h, pred_o, act_o = forward_prop(w_i_h, b_i_h, w_h_o, b_h_o, inputs)
        cost = sum([log_loss(a, t) for a, t in zip(act_o, data.targets)]) / len(act_o)
        print(f"epoch:{i} cost:{cost:.4f}")
        w_h_o_d, b_h_o_d, w_i_h_d, b_i_h_d = backward_prop(pred_h, act_h, act_o, w_h_o, inputs, targets)
        w_i_h, b_i_h, w_h_o, b_h_o = update_params(w_i_h, b_i_h, w_h_o, b_h_o, w_i_h_d, b_i_h_d, w_h_o_d, b_h_o_d, learning_rate, 2,
                  4, 3, inputs)
        # if i % 10 == 0:
        #     print("Iteration: ", i)
        #     predictions = get_predictions(A2)
        #     print(get_accuracy(predictions, Y))
    return w_i_h, b_i_h, w_h_o, b_h_o

w1,b1,w2,b2 = gradient_descent(w_i_h,b_i_h,w_h_o,b_h_o,data.inputs,data.targets,learning_rate,epochs)

pred_h = [[sum([w * a for w,a in zip(weights,inp)]) + bias for weights,bias in zip(w1,b1)] for inp in data.test_inputs]
act_h = [[max(0,p) for p in pred] for pred in pred_h]
pred_o = [[sum([w * a for w,a in zip(weights,act)]) + bias for weights, bias in zip(w2,b2)] for act in act_h]
act_o = [softmax(predictions) for predictions in pred_o]
correct = 0
for a,t in zip(act_o,data.test_targets):
    # ma_neuron = a.index(max(a))
    # ma_target = t.index(max(t))
    if a.index(max(a)) == t.index(max(t)):
        correct += 1
    # else:
    #     print(f"degit:{ma_target}, guessed:{ma_neuron}")

print(f"Correct: {correct}/{len(act_o)} ({correct/len(act_o):%})")