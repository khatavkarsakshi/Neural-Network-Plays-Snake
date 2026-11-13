import numpy as np


class NeuralNetwork:
    """
    11 -> 64 (ReLU) -> 3 (linear Q-values), pure NumPy.

    forward:   h = max(0, x @ W1 + b1)      Q = h @ W2 + b2
    backprop:  dQ  = 2 * (Q - y) / n
               dW2 = h.T @ dQ
               dh  = (dQ @ W2.T) * (h > 0)
               dW1 = x.T @ dh
    Weights are updated with Adam.
    """

    def __init__(self, input_size=11, hidden_size=64, output_size=3, lr=0.001, seed=None):
        rng = np.random.default_rng(seed)
        self.lr = lr

        # He initialisation (good for ReLU)
        self.params = {
            "W1": rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size),
            "b1": np.zeros((1, hidden_size)),
            "W2": rng.standard_normal((hidden_size, output_size)) * np.sqrt(2.0 / hidden_size),
            "b2": np.zeros((1, output_size)),
        }

        # Adam state
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

        self.cache = {}

    def forward(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        z1 = x @ self.params["W1"] + self.params["b1"]
        a1 = np.maximum(0.0, z1)                           # ReLU
        out = a1 @ self.params["W2"] + self.params["b2"]   # linear Q-values
        self.cache = {"x": x, "z1": z1, "a1": a1}
        return out

    def predict(self, x):
        return self.forward(x)

    def backward(self, out, targets):
        """Backprop for MSE loss using the cache of the last forward pass."""
        n = out.shape[0]
        x, z1, a1 = self.cache["x"], self.cache["z1"], self.cache["a1"]

        d_out = 2.0 * (out - targets) / n
        grads = {}
        grads["W2"] = a1.T @ d_out
        grads["b2"] = d_out.sum(axis=0, keepdims=True)

        d_a1 = d_out @ self.params["W2"].T
        d_z1 = d_a1 * (z1 > 0)
        grads["W1"] = x.T @ d_z1
        grads["b1"] = d_z1.sum(axis=0, keepdims=True)
        return grads

    def _update(self, grads):
        self.t += 1
        for k in self.params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def fit_batch(self, out, targets):
        """Backprop + Adam step. `out` must come from the latest forward() call."""
        loss = float(np.mean(np.sum((out - targets) ** 2, axis=1)))
        grads = self.backward(out, targets)
        self._update(grads)
        return loss

    def train_step(self, states, targets):
        targets = np.atleast_2d(np.asarray(targets, dtype=np.float64))
        out = self.forward(states)
        return self.fit_batch(out, targets)

    def copy_from(self, other):
        for k in self.params:
            self.params[k] = other.params[k].copy()

    def save(self, path="snake_model_last.npz"):
        np.savez(path, **self.params)

    def load(self, path="snake_model_last.npz"):
        data = np.load(path)
        for k in self.params:
            self.params[k] = data[k]