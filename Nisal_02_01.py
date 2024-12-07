# Nisal, Prajakta
# 1002_174_111
# 2024_10_01
# Assignment_02_01


# def svm_loss(y_true, y_pred):
#     margin = 1 - y_true * y_pred
#     loss = torch.clamp(margin, min=0)
#     return torch.mean(loss)


# def cross_entropy_loss(y_true, y_pred):
#     loss = -torch.log(torch.exp(y_pred) / torch.sum(torch.exp(y_pred)))
#     return torch.mean(loss)

import numpy as np
import torch


def mse_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def svm_loss(y_true, y_pred):
    margin = 1 - y_true * y_pred
    loss = torch.clamp(margin, min=0)
    return torch.mean(loss)


def cross_entropy_loss(y_true, y_pred):
    exp = torch.exp(y_pred)
    row_sums = torch.sum(exp, 1, keepdim=True)
    softmax = exp / row_sums
    prod_sm = torch.mul(y_true, softmax)
    log_sm = -torch.log(softmax)
    loss = torch.sum(log_sm, 1, keepdim=True)
    mean_loss = torch.mean(loss)
    return mean_loss


def multi_layer_nn_torch(
    x_train,
    y_train,
    layers,
    activations,
    alpha=0.01,
    batch_size=32,
    epochs=0,
    loss_func="mse",
    val_split=(0.8, 1.0),
    seed=7321,
):

    np.random.seed(seed)

    num_samples = x_train.shape[0]
    start_idx = int(np.floor(val_split[0] * num_samples))
    end_idx = int(np.floor(val_split[1] * num_samples))

    x_val, y_val = x_train[start_idx:end_idx], y_train[start_idx:end_idx]
    x_train, y_train = x_train[:start_idx], y_train[:start_idx]

    # Convert numpy arrays to PyTorch tensors
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).float()

    # Initialize weights
    if isinstance(layers[0], int):
        input_size = x_train.shape[1]
        layers = [input_size] + layers
        weights = []

        for i in range(1, len(layers)):
            np.random.seed(seed)
            w_np = np.random.randn(layers[i - 1] + 1, layers[i])
            w = torch.tensor(w_np, dtype=torch.float32, requires_grad=True)
            weights.append(w)
    else:
        weights = [
            torch.tensor(w, dtype=torch.float32, requires_grad=True) for w in layers
        ]

    print("Weightss Initialized")

    # Activation functions
    activation_functions = {
        "linear": lambda x: x,
        "sigmoid": torch.sigmoid,
        "relu": torch.relu,
    }

    loss_function = {"mse": mse_loss, "svm": svm_loss, "ce": cross_entropy_loss}[
        loss_func.lower()
    ]

    error_history = []
    if epochs == 0:
        return [w.detach().numpy() for w in weights], error_history, np.array([])

    # Training loop
    for epoch in range(epochs):
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            # Forward pass
            activations_list = [x_batch]
            # print(x_batch.shape)
            for j in range(len(layers) - 1):
                ones = torch.ones((x_batch.shape[0], 1))
                act = activations_list[-1]
                A_input = torch.cat([ones, act], axis=1)
                # print(f"a_in : {A_input.shape}")
                # print(f"weights[{j}] : {weights[j].shape}")
                Z = torch.mm(A_input, weights[j])
                A = activation_functions[activations[j]](Z)
                activations_list.append(A)

            loss = loss_function(y_batch, activations_list[-1])

            loss.backward()

            with torch.no_grad():
                for j in range(len(layers) - 1):
                    weights[j] -= alpha * weights[j].grad
                    weights[j].grad.zero_()

        with torch.no_grad():
            activations_val = [x_val]
            for j in range(len(layers) - 1):
                A_input = torch.cat(
                    [torch.ones((x_val.shape[0], 1)), activations_val[-1]], axis=1
                )
                Z = torch.mm(A_input, weights[j])
                A = activation_functions[activations[j]](Z)
                # print(f'A::{A}')
                activations_val.append(A)
            val_loss = torch.mean(torch.abs(y_val - activations_val[-1]))
            error_history.append(val_loss.item())

    weights_np = [w.detach().numpy() for w in weights]
    final_output = activations_val[-1].detach().numpy()

    return weights_np, error_history, final_output


def get_data():
    x_train = np.array(
        [
            [0.685938, -0.5756752],
            [0.944493, -0.02803439],
            [0.9477775, 0.59988844],
            [0.20710745, -0.12665261],
            [-0.08198895, 0.22326154],
            [-0.77471393, -0.73122877],
            [-0.18502127, 0.32624513],
            [-0.03133733, -0.17500992],
            [0.28585237, -0.01097354],
            [-0.19126464, 0.06222228],
            [-0.0303282, -0.16023481],
            [-0.34069192, -0.8288299],
            [-0.20600465, 0.09318836],
            [0.29411194, -0.93214977],
            [-0.7150941, 0.74259764],
            [0.13344735, 0.17136675],
            [0.31582892, 1.0810335],
            [-0.22873795, 0.98337173],
            [-0.88140666, 0.05909261],
            [-0.21215424, -0.05584779],
        ],
        dtype=np.float32,
    )

    y_train = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    return x_train, y_train


np.random.seed(7321)
x_train, y_train = get_data()
print(x_train.shape)
weight_mat, error, output = multi_layer_nn_torch(
    x_train=x_train,
    y_train=y_train,
    layers=[8, 6, 7, 5, 3, 1, 9, 2],
    activations=["relu", "relu", "relu", "relu", "relu", "relu", "relu", "linear"],
    alpha=0.01,
    batch_size=32,
    epochs=2,
    loss_func="ce",
    seed=7321,
)
