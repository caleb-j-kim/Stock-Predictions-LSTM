import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv('https://raw.githubusercontent.com/DilonSok/ML-Datasets/refs/heads/main/aapl_us_d.csv')
data = data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Visualize data
# plt.plot(data['Close'])
# plt.xlabel('Date')
# plt.ylabel('Close Price (USD)')
# plt.title('Stock Price History')
# plt.show()

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Prepare sequences
sequence_length = 120
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], sequence_length, 1))

# Split data into train and test sets
split_ratio = 0.8
train_size = int(len(X) * split_ratio)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define LSTM Cell with Gradient Clipping and L2 Regularization
class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.1
        self.bf = np.zeros((hidden_dim, 1))
        self.Wi = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.1
        self.bi = np.zeros((hidden_dim, 1))
        self.Wc = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.1
        self.bc = np.zeros((hidden_dim, 1))
        self.Wo = np.random.randn(hidden_dim, input_dim + hidden_dim) * 0.1
        self.bo = np.zeros((hidden_dim, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def clip_gradients(self, gradient, clip_value=1.0):
        return np.clip(gradient, -clip_value, clip_value)

    def forward(self, x, h_prev, c_prev):
        concat = np.vstack((h_prev, x))
        f = self.sigmoid(self.Wf @ concat + self.bf)
        i = self.sigmoid(self.Wi @ concat + self.bi)
        c_hat = self.tanh(self.Wc @ concat + self.bc)
        c = f * c_prev + i * c_hat
        o = self.sigmoid(self.Wo @ concat + self.bo)
        h = o * self.tanh(c)
        cache = (h, c, f, i, c_hat, o, c_prev, h_prev, x)
        return h, c, cache

    def backward(self, dh_next, dc_next, cache):
        h, c, f, i, c_hat, o, c_prev, h_prev, x = cache
        concat = np.vstack((h_prev, x))

        do = dh_next * np.tanh(c)
        do = do * o * (1 - o)

        dc = dc_next + dh_next * o * (1 - np.tanh(c) ** 2)
        dc_hat = dc * i * (1 - c_hat ** 2)
        di = dc * c_hat * i * (1 - i)
        df = dc * c_prev * f * (1 - f)

        dWf = df @ concat.T
        dWi = di @ concat.T
        dWc = dc_hat @ concat.T
        dWo = do @ concat.T

        dbf = np.sum(df, axis=1, keepdims=True)
        dbi = np.sum(di, axis=1, keepdims=True)
        dbc = np.sum(dc_hat, axis=1, keepdims=True)
        dbo = np.sum(do, axis=1, keepdims=True)

        d_concat = self.Wf.T @ df + self.Wi.T @ di + self.Wc.T @ dc_hat + self.Wo.T @ do
        dh_prev = d_concat[:self.hidden_dim, :]
        dc_prev = f * dc

        # Apply gradient clipping
        dWf = self.clip_gradients(dWf)
        dbf = self.clip_gradients(dbf)
        dWi = self.clip_gradients(dWi)
        dbi = self.clip_gradients(dbi)
        dWc = self.clip_gradients(dWc)
        dbc = self.clip_gradients(dbc)
        dWo = self.clip_gradients(dWo)
        dbo = self.clip_gradients(dbo)

        return dh_prev, dc_prev, dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo

# Define LSTM Model
class LSTMModel:
    def __init__(self, input_dim, hidden_dim, output_dim, sequence_length):
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.cell = LSTMCell(input_dim, hidden_dim)
        self.Wy = np.random.randn(output_dim, hidden_dim) * 0.1
        self.by = np.zeros((output_dim, 1))

    def forward(self, X):
        batch_size = X.shape[0]
        h = np.zeros((self.hidden_dim, batch_size))
        c = np.zeros((self.hidden_dim, batch_size))
        self.caches = []
        for t in range(self.sequence_length):
            x_t = X[:, t, :].T
            h, c, cache_t = self.cell.forward(x_t, h, c)
            self.caches.append(cache_t)
        y = self.Wy @ h + self.by
        return y.T

    def backward(self, dy, learning_rate, lambda_reg):
        dy = dy.T
        dWy = dy @ self.caches[-1][0].T + lambda_reg * self.Wy  # L2 regularization for Wy
        dby = np.sum(dy, axis=1, keepdims=True)

        dh_next = self.Wy.T @ dy
        dc_next = np.zeros_like(dh_next)

        dWf = np.zeros_like(self.cell.Wf)
        dbf = np.zeros_like(self.cell.bf)
        dWi = np.zeros_like(self.cell.Wi)
        dbi = np.zeros_like(self.cell.bi)
        dWc = np.zeros_like(self.cell.Wc)
        dbc = np.zeros_like(self.cell.bc)
        dWo = np.zeros_like(self.cell.Wo)
        dbo = np.zeros_like(self.cell.bo)

        for t in reversed(range(self.sequence_length)):
            cache_t = self.caches[t]
            dh_next, dc_next, dWf_t, dbf_t, dWi_t, dbi_t, dWc_t, dbc_t, dWo_t, dbo_t = self.cell.backward(
                dh_next, dc_next, cache_t)

            dWf += dWf_t + lambda_reg * self.cell.Wf  # L2 regularization for Wf
            dbf += dbf_t
            dWi += dWi_t + lambda_reg * self.cell.Wi  # L2 regularization for Wi
            dbi += dbi_t
            dWc += dWc_t + lambda_reg * self.cell.Wc  # L2 regularization for Wc
            dbc += dbc_t
            dWo += dWo_t + lambda_reg * self.cell.Wo  # L2 regularization for Wo
            dbo += dbo_t

        self.Wy -= learning_rate * self.cell.clip_gradients(dWy)
        self.by -= learning_rate * self.cell.clip_gradients(dby)

        self.cell.Wf -= learning_rate * dWf
        self.cell.bf -= learning_rate * dbf
        self.cell.Wi -= learning_rate * dWi
        self.cell.bi -= learning_rate * dbi
        self.cell.Wc -= learning_rate * dWc
        self.cell.bc -= learning_rate * dbc
        self.cell.Wo -= learning_rate * dWo
        self.cell.bo -= learning_rate * dbo

# Loss function and metrics
def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def root_mean_squared_error(y_pred, y_true):
    return np.sqrt(mean_squared_error(y_pred, y_true))

# Training loop with metrics
def train(model, X_train, y_train, X_test, y_test, epochs, learning_rate, batch_size, lambda_reg):
    num_samples = X_train.shape[0]
    for epoch in range(epochs):
        total_train_loss = 0

        # Shuffle the data
        permutation = np.random.permutation(num_samples)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        for i in range(0, num_samples, batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            y_pred = model.forward(X_batch)
            y_true = y_batch.reshape(-1, 1)

            # Calculate training loss
            batch_loss = mean_squared_error(y_pred, y_true)
            total_train_loss += batch_loss * X_batch.shape[0]

            dy = y_pred - y_true
            model.backward(dy, learning_rate, lambda_reg)

        # Calculate testing metrics for the epoch
        y_test_pred = np.array([model.forward(X_test[i:i+1])[0, 0] for i in range(X_test.shape[0])]).reshape(-1, 1)
        test_loss = mean_squared_error(y_test_pred, y_test.reshape(-1, 1))
        test_rmse = np.sqrt(test_loss)
        ss_res = np.sum((y_test.reshape(-1) - y_test_pred.reshape(-1)) ** 2)
        ss_tot = np.sum((y_test.reshape(-1) - np.mean(y_test)) ** 2)
        test_r2 = 1 - (ss_res / ss_tot)

        # Output metrics to console
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_train_loss / num_samples:.4f}, "
              f"Train RMSE: {np.sqrt(total_train_loss / num_samples):.4f}, "
              f"Test Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")

    return total_train_loss / num_samples, np.sqrt(total_train_loss / num_samples), test_loss, test_rmse, test_r2

# Parameter grid for systematic testing
batch_sizes = [16, 32]
epochs_list = [10, 20]
learning_rates = [0.01, 0.05]
lambda_regs = [0.00001, 0.0001]
hidden_dims = [50, 100]
sequence_lengths = [60, 120]

# Logging function
def log_metrics(batch_size, epochs, learning_rate, lambda_reg, hidden_dim, sequence_length, metrics):
    with open('training_log.txt', 'a') as f:
        f.write(f"Batch Size: {batch_size}, Epochs: {epochs}, Learning Rate: {learning_rate}, "
                f"Lambda: {lambda_reg}, Hidden Dim: {hidden_dim}, Sequence Length: {sequence_length}\n"
                f"Training Loss: {metrics[0]:.4f}, Training RMSE: {metrics[1]:.4f}, "
                f"Testing Loss: {metrics[2]:.4f}, Testing RMSE: {metrics[3]:.4f}, Testing R²: {metrics[4]:.4f}\n\n")

# Track best and worst metrics
best_params = None
worst_params = None
best_rmse = float('inf')
worst_rmse = 0

# Iterate through parameter combinations
for batch_size, epochs, learning_rate, lambda_reg, hidden_dim, sequence_length in itertools.product(
        batch_sizes, epochs_list, learning_rates, lambda_regs, hidden_dims, sequence_lengths):

    # Prepare sequences based on sequence length
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], sequence_length, 1))

    # Split data into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = LSTMModel(input_dim=1, hidden_dim=hidden_dim, output_dim=1, sequence_length=sequence_length)
    metrics = train(model, X_train, y_train, X_test, y_test, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, lambda_reg=lambda_reg)
    log_metrics(batch_size, epochs, learning_rate, lambda_reg, hidden_dim, sequence_length, metrics)

    # Update best and worst metrics
    if metrics[3] < best_rmse:
        best_rmse = metrics[3]
        best_params = (batch_size, epochs, learning_rate, lambda_reg, hidden_dim, sequence_length, metrics)

    if metrics[3] > worst_rmse:
        worst_rmse = metrics[3]
        worst_params = (batch_size, epochs, learning_rate, lambda_reg, hidden_dim, sequence_length, metrics)

# Log best and worst parameter sets
with open('training_log.txt', 'a') as f:
    f.write("\nBest Parameter Set:\n")
    f.write(f"Batch Size: {best_params[0]}, Epochs: {best_params[1]}, Learning Rate: {best_params[2]}, "
            f"Lambda: {best_params[3]}, Hidden Dim: {best_params[4]}, Sequence Length: {best_params[5]}\n"
            f"Training Loss: {best_params[6][0]:.4f}, Training RMSE: {best_params[6][1]:.4f}, "
            f"Testing Loss: {best_params[6][2]:.4f}, Testing RMSE: {best_params[6][3]:.4f}, Testing R²: {best_params[6][4]:.4f}\n\n")

    f.write("Worst Parameter Set:\n")
    f.write(f"Batch Size: {worst_params[0]}, Epochs: {worst_params[1]}, Learning Rate: {worst_params[2]}, "
            f"Lambda: {worst_params[3]}, Hidden Dim: {worst_params[4]}, Sequence Length: {worst_params[5]}\n"
            f"Training Loss: {worst_params[6][0]:.4f}, Training RMSE: {worst_params[6][1]:.4f}, "
            f"Testing Loss: {worst_params[6][2]:.4f}, Testing RMSE: {worst_params[6][3]:.4f}, Testing R²: {worst_params[6][4]:.4f}\n")
