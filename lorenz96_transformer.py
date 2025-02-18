"""
Implementing a Transformer-based neural network for Lorenz '96 Y variable prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import math
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

F_FORCING = 20
h = 1
b = 10
c = 10
K = 8
J = 8

class Lorenz96Dataset(Dataset):
    """Custom Dataset for Lorenz '96 data"""
    def __init__(self, X_data, Y_data):
        self.X_data = torch.FloatTensor(X_data)
        self.Y_data = torch.FloatTensor(Y_data)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return self.X_data[idx], self.Y_data[idx]

class CircularEncoding(nn.Module):
    """Circular encoding to capture the periodic boundary conditions"""
    def __init__(self, input_size=K):
        super().__init__()
        angles = 2 * math.pi * torch.arange(input_size) / input_size
        self.register_buffer('sin_encoding', torch.sin(angles))
        self.register_buffer('cos_encoding', torch.cos(angles))

    def forward(self, x):
        batch_size = x.size(0)
        sin_features = self.sin_encoding.expand(batch_size, -1)
        cos_features = self.cos_encoding.expand(batch_size, -1)

        return torch.cat([x, sin_features, cos_features], dim=1)

class Lorenz96TransformerNet(nn.Module):
    """Transformer-based neural network for Lorenz96 Y variable prediction"""
    def __init__(self, input_size=K, hidden_sizes=[256, 512, 512, 512, 512, 512, 256], output_size=K*J,
                 dropout_rate=0.2, use_attention=True):
        super().__init__()

        self.use_attention = use_attention

        self.circular_encoding = CircularEncoding(input_size)
        enhanced_input_size = input_size * 3

        self.input_layer = nn.Sequential(
            nn.Linear(enhanced_input_size, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_sizes[0],
                num_heads=4,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_sizes[0])

        self.hidden_layers = nn.ModuleList([
            ResidualBlock(hidden_sizes[i], hidden_sizes[i+1], dropout_rate)
            for i in range(len(hidden_sizes) - 1)
        ])

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_sizes[-1], output_size),
            nn.LayerNorm(output_size)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Kaiming initialization"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.circular_encoding(x)

        x = self.input_layer(x)

        if self.use_attention:
            x_reshaped = x.unsqueeze(1)
            attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
            x = x + attn_out.squeeze(1)
            x = self.attention_norm(x)

        for layer in self.hidden_layers:
            x = layer(x)

        return self.output_layer(x)

    def get_l2_regularization(self):
        """Calculate L2 regularization term for all parameters"""
        return sum(torch.norm(param) for param in self.parameters())

def train_model_M1(model, X_train, Y_train, X_val, Y_val,
                epochs=200, batch_size=64, initial_lr=0.001,
                weight_decay=1e-5, early_stopping_patience=15,
                scheduler_patience=5, scheduler_factor=0.5):
    """Training function specifically for model M1"""

    train_dataset = Lorenz96Dataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr,
                                weight_decay=weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor,
        patience=scheduler_patience, verbose=True
    )
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'learning_rates': [], 'y_sum_errors': []}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_y_sum_error = 0

        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                Y_pred = model(batch_X)
                pred_loss = criterion(Y_pred, batch_Y)

                Y_pred_reshaped = Y_pred.reshape(-1, K, J)
                Y_true_reshaped = batch_Y.reshape(-1, K, J)
                sum_pred = torch.sum(Y_pred_reshaped, dim=2)
                sum_true = torch.sum(Y_true_reshaped, dim=2)
                sum_loss = criterion(sum_pred, sum_true)

                loss = pred_loss + 0.1 * sum_loss

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            total_y_sum_error += sum_loss.item()

        model.eval()
        with torch.no_grad():
            Y_pred = model(torch.FloatTensor(X_val))
            val_pred_loss = criterion(Y_pred, torch.FloatTensor(Y_val))

            Y_pred_reshaped = Y_pred.reshape(-1, K, J)
            Y_val_reshaped = torch.FloatTensor(Y_val).reshape(-1, K, J)
            val_sum_pred = torch.sum(Y_pred_reshaped, dim=2)
            val_sum_true = torch.sum(Y_val_reshaped, dim=2)
            val_sum_loss = criterion(val_sum_pred, val_sum_true)

            val_loss = val_pred_loss + 0.1 * val_sum_loss

        avg_train_loss = total_train_loss / len(train_loader)
        avg_y_sum_error = total_y_sum_error / len(train_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss.item())
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        history['y_sum_errors'].append(avg_y_sum_error)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {val_loss.item():.6f}, '
                  f'Y Sum Error: {avg_y_sum_error:.6f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

    return best_model, history

def fine_tune_model(model, X_data, Y_data, F_new=24, epochs=50):
    """Fine-tune model for new forcing parameter F"""
    X_new, Y_new, _ = generate_training_data(num_timesteps=1000, F=F_new)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_new, Y_new.reshape(len(X_new), -1),
        test_size=0.2, random_state=42
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=42
    )

    model_ft, history_ft = train_model_M1(
        model,
        X_train, Y_train,
        X_val, Y_val,
        epochs=epochs,
        initial_lr=0.0001,
        batch_size=32
    )

    return model_ft, history_ft, (X_test, Y_test)

def stabilize_hybrid_system(model, X0, t, max_attempts=5):
    """Enhanced system stabilization with multiple strategies"""
    best_solution = None
    best_metrics = None
    best_stability = float('inf')

    strategies = [
        {'smoothing': 0.0, 'dt_factor': 1.0},
        {'smoothing': 0.05, 'dt_factor': 1.0},
        {'smoothing': 0.1, 'dt_factor': 0.5},
        {'smoothing': 0.0, 'dt_factor': 0.5},
    ]

    def hybrid_with_smoothing(state, t, model, smooth_factor, prev_Y=None):
        """Hybrid system with stability controls"""
        X = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            Y_pred = model(X).numpy().reshape(K, J)

        if prev_Y is not None:
            Y_pred = Y_pred * (1 - smooth_factor) + prev_Y * smooth_factor

        dX = np.zeros(K)
        for k in range(K):
            dX[k] = (state[(k+1)%K] - state[k-2]) * state[k-1] - state[k] + F_FORCING
            dX[k] -= h * c / b * np.sum(Y_pred[k])

        return dX, Y_pred

    print("\nAttempting system stabilization...")
    for strategy in strategies:
        prev_Y = None
        try:
            dt = (t[1] - t[0]) * strategy['dt_factor']
            t_adjusted = np.arange(t[0], t[-1], dt)

            states = [X0]
            for i in range(1, len(t_adjusted)):
                dt = t_adjusted[i] - t_adjusted[i-1]
                current_state = states[-1]

                k1, Y1 = hybrid_with_smoothing(current_state, t_adjusted[i], model,
                                             strategy['smoothing'], prev_Y)
                k2, Y2 = hybrid_with_smoothing(current_state + dt*k1/2, t_adjusted[i], model,
                                             strategy['smoothing'], Y1)
                k3, Y3 = hybrid_with_smoothing(current_state + dt*k2/2, t_adjusted[i], model,
                                             strategy['smoothing'], Y2)
                k4, Y4 = hybrid_with_smoothing(current_state + dt*k3, t_adjusted[i], model,
                                             strategy['smoothing'], Y3)

                new_state = current_state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
                states.append(new_state)
                prev_Y = Y4

            hybrid_solution = np.array(states)

            if len(t_adjusted) != len(t):
                from scipy.interpolate import interp1d
                f = interp1d(t_adjusted, hybrid_solution, axis=0)
                hybrid_solution = f(t)

            original_solution = odeint(
                lorenz96_full,
                np.concatenate([X0, np.random.randn(K*J)]),
                t
            )

            differences = hybrid_solution - original_solution[:, :K]
            metrics = {
                'mean_abs_diff': np.mean(np.abs(differences)),
                'max_abs_diff': np.max(np.abs(differences)),
                'std_diff': np.std(differences),
                'normalized_rmse': np.sqrt(np.mean(differences**2)) / np.std(original_solution[:, :K])
            }

            if metrics['normalized_rmse'] < best_stability:
                best_stability = metrics['normalized_rmse']
                best_solution = hybrid_solution.copy()
                best_metrics = metrics.copy()

            print(f"Strategy (smoothing={strategy['smoothing']:.2f}, "
                  f"dt_factor={strategy['dt_factor']:.1f}) - "
                  f"NRMSE: {metrics['normalized_rmse']:.6f}")

        except Exception as e:
            print(f"Strategy failed with error: {str(e)}")
            continue

    if best_solution is None:
        raise RuntimeError("Failed to stabilize system with any strategy")

    return best_solution, best_metrics

def lorenz96_full(state, t, F=F_FORCING):
    """Full Lorenz '96 system with X and Y variables"""
    X = state[:K]
    Y = state[K:].reshape(K, J)

    dX = np.zeros(K)
    dY = np.zeros((K, J))

    for k in range(K):
        dX[k] = (X[(k+1)%K] - X[k-2]) * X[k-1] - X[k] + F
        dX[k] -= h * c / b * np.sum(Y[k])

    for k in range(K):
        for j in range(J):
            dY[k,j] = -c * b * Y[k,(j+1)%J] * (Y[k,(j+2)%J] - Y[k,j-1]) \
                      - c * Y[k,j] + h * c / b * X[k]

    return np.concatenate([dX, dY.flatten()])

def lorenz96_hybrid(state, t, model):
    """Hybrid system using neural network for Y predictions"""
    X = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        Y_pred = model(X).numpy().reshape(K, J)

    dX = np.zeros(K)
    for k in range(K):
        dX[k] = (state[(k+1)%K] - state[k-2]) * state[k-1] - state[k] + F_FORCING
        dX[k] -= h * c / b * np.sum(Y_pred[k])

    return dX

def generate_training_data(num_timesteps=1000, dt=0.01, F=F_FORCING):
    """Generate training data from full system integration"""
    X0 = F + np.random.randn(K)
    Y0 = np.random.randn(K, J).flatten()
    state0 = np.concatenate([X0, Y0])

    t = np.arange(0, num_timesteps * dt, dt)
    solution = odeint(lorenz96_full, state0, t, args=(F,))

    X_data = solution[:, :K]
    Y_data = solution[:, K:].reshape(-1, K, J)
    Y_sums = np.sum(Y_data, axis=2)

    return X_data, Y_data, Y_sums, t

class LorenzVisualizer:
    """Visualization tools for Lorenz96 system analysis"""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def plot_system_evolution(self, t, X_data, Y_data, title="Full System Evolution"):
        """Plot the evolution of X and Y variables over time with one pair per subplot"""
        num_vars = X_data.shape[1]
        rows = min(4, num_vars)
        cols = int(np.ceil(num_vars / rows))
        fig = plt.figure(figsize=(5*cols, 3*rows))

        Y_sums = np.sum(Y_data.reshape(len(X_data), -1, J), axis=2)

        for k in range(num_vars):
            ax = plt.subplot(rows, cols, k+1)
            color = plt.cm.tab10(k/10)

            ax.plot(t, X_data[:, k], color=color, label='X')
            ax.plot(t, Y_sums[:, k], '--', color=color, label='ΣY')

            ax.set_title(f'Variable {k+1}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_training_history(self, history):
        """Plot training and validation losses"""
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history['learning_rates'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_prediction_comparison(self, y_true, y_pred, title="Model Predictions"):
        """Plot prediction analysis with one pair per subplot"""
        fig = plt.figure(figsize=(15, 10))

        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax1.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.1)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        ax1.set_title('True vs Predicted Values')
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        ax1.grid(True)

        ax2 = plt.subplot2grid((2, 2), (0, 1))
        residuals = y_pred.flatten() - y_true.flatten()
        ax2.hist(residuals, bins=50, density=True)
        ax2.set_title('Residuals Distribution')
        ax2.set_xlabel('Residual Value')
        ax2.set_ylabel('Density')
        ax2.grid(True)

        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        abs_errors = np.abs(y_pred - y_true)
        ax3.boxplot([errors for errors in abs_errors[:100].T])
        ax3.set_title('Error Distribution Over Time')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Absolute Error')
        ax3.grid(True)

        plt.suptitle('Diagnostic Plots')
        plt.tight_layout()
        plt.show()

        num_series = min(8, y_true.shape[1])
        rows = min(4, num_series)
        cols = int(np.ceil(num_series / rows))

        fig = plt.figure(figsize=(5*cols, 3*rows))

        for i in range(num_series):
            ax = plt.subplot(rows, cols, i+1)
            color = plt.cm.tab10(i/10)

            ax.plot(y_true[:100, i], color=color, label='True')
            ax.plot(y_pred[:100, i], '--', color=color, label='Predicted')

            ax.set_title(f'Series {i+1}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)

        plt.suptitle('Time Series Comparisons (First 100 Steps)')
        plt.tight_layout()
        plt.show()

        print("\nModel Performance Metrics:")
        print(f"Mean Squared Error: {mean_squared_error(y_true, y_pred):.6f}")
        print(f"Mean Absolute Error: {np.mean(np.abs(residuals)):.6f}")
        print(f"Residuals Standard Deviation: {np.std(residuals):.6f}")
        print(f"R² Score: {1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2):.6f}")

    def plot_3d_comparison(self, true_data, pred_data, hybrid_data, title="System Comparison"):
        """Plot 3D phase space comparison"""
        fig = plt.figure(figsize=(15, 5))

        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(true_data[:, 0], true_data[:, 1], true_data[:, 2],
                color='blue', alpha=0.7)
        ax1.set_title('True System')

        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot(pred_data[:, 0], pred_data[:, 1], pred_data[:, 2],
                color='red', alpha=0.7)
        ax2.set_title('Predicted System')

        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot(hybrid_data[:, 0], hybrid_data[:, 1], hybrid_data[:, 2],
                color='green', alpha=0.7)
        ax3.set_title('Hybrid System')

        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('X₁')
            ax.set_ylabel('X₂')
            ax.set_zlabel('X₃')
            ax.view_init(elev=30, azim=45)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

def analyze_stability(t, hybrid_solution, original_solution):
    """Analyze and visualize system stability with one pair per subplot"""
    differences = hybrid_solution - original_solution[:, :K]

    metrics = {
        'mean_abs_diff': np.mean(np.abs(differences)),
        'max_abs_diff': np.max(np.abs(differences)),
        'std_diff': np.std(differences),
        'relative_error': np.mean(np.abs(differences) / np.abs(original_solution[:, :K]))
    }

    num_vars = hybrid_solution.shape[1]
    rows = min(4, num_vars)
    cols = int(np.ceil(num_vars / rows))

    fig = plt.figure(figsize=(5*cols, 3*rows))

    for k in range(num_vars):
        ax = plt.subplot(rows, cols, k+1)
        color = plt.cm.tab10(k/10)

        ax.plot(t, hybrid_solution[:, k], color=color, label='Hybrid')
        ax.plot(t, original_solution[:, k], '--', color=color, label='Original')

        ax.set_title(f'Variable {k+1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    plt.suptitle('System Evolution Comparison')
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(5*cols, 3*rows))

    for k in range(num_vars):
        ax = plt.subplot(rows, cols, k+1)
        ax.plot(t, differences[:, k])
        ax.set_title(f'Differences for Variable {k+1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Difference')
        ax.grid(True)

    plt.suptitle('System Differences')
    plt.tight_layout()
    plt.show()

    print("\nStability Analysis:")
    print(f"Mean Absolute Difference: {metrics['mean_abs_diff']:.6f}")
    print(f"Maximum Absolute Difference: {metrics['max_abs_diff']:.6f}")
    print(f"Standard Deviation of Differences: {metrics['std_diff']:.6f}")
    print(f"Mean Relative Error: {metrics['relative_error']:.6f}")

    return metrics

def add_temporal_features(X_data):
    """Add temporal features to X data"""
    X_diff = np.diff(X_data, axis=0)
    X_diff = np.vstack([np.zeros((1, X_data.shape[1])), X_diff])

    X_momentum = np.diff(X_diff, axis=0)
    X_momentum = np.vstack([np.zeros((1, X_data.shape[1])), X_momentum])

    window_sizes = [3, 5, 7]
    X_mas = []
    for w in window_sizes:
        ma = np.array([np.convolve(X_data[:, i], np.ones(w)/w, mode='same')
                      for i in range(X_data.shape[1])]).T
        X_mas.append(ma)

    return np.hstack([X_data, X_diff, X_momentum] + X_mas)

def add_spatial_features(X_data):
    """Add spatial relationship features"""
    K = X_data.shape[1]

    X_spatial_diff = np.zeros_like(X_data)
    for i in range(K):
        X_spatial_diff[:, i] = X_data[:, (i+1)%K] - X_data[:, i-1]

    X_curvature = np.zeros_like(X_data)
    for i in range(K):
        X_curvature[:, i] = X_data[:, (i+1)%K] + X_data[:, i-1] - 2*X_data[:, i]

    return np.hstack([X_data, X_spatial_diff, X_curvature])

def add_physics_features(X_data, F=F_FORCING):
    """Add physics-informed features based on Lorenz96 equations"""
    K = X_data.shape[1]

    X_forcing = np.full_like(X_data, F)

    X_advection = np.zeros_like(X_data)
    for k in range(K):
        X_advection[:, k] = X_data[:, (k+1)%K] * X_data[:, k-1]

    X_energy = X_data ** 2

    return np.hstack([X_data, X_forcing, X_advection, X_energy])

def add_statistical_features(X_data):
    """Add statistical features"""
    window_size = 5
    X_roll_mean = pd.DataFrame(X_data).rolling(window=window_size, min_periods=1).mean().values
    X_roll_std = pd.DataFrame(X_data).rolling(window=window_size, min_periods=1).std().values
    X_roll_skew = pd.DataFrame(X_data).rolling(window=window_size, min_periods=1).skew().values

    X_global_mean = np.mean(X_data, axis=1, keepdims=True)
    X_global_std = np.std(X_data, axis=1, keepdims=True)

    return np.hstack([X_data, X_roll_mean, X_roll_std, X_roll_skew,
                      X_global_mean, X_global_std])

def prepare_enhanced_data(X_data):
    """Combine all feature engineering techniques"""
    X_enhanced = X_data.copy()
    X_enhanced = add_temporal_features(X_enhanced)
    X_enhanced = add_spatial_features(X_enhanced)
    X_enhanced = add_physics_features(X_enhanced)
    X_enhanced = add_statistical_features(X_enhanced)
    X_enhanced = np.nan_to_num(X_enhanced, nan=0.0)

    return X_enhanced

def main():
    """Main function demonstrating the usage of the Lorenz96 system"""
    visualizer = LorenzVisualizer()

    print("Generating training data...")
    X_data, Y_data, Y_sums, t = generate_training_data(num_timesteps=1000)

    visualizer.plot_system_evolution(t, X_data, Y_data, f"Lorenz96 System (F={F_FORCING})")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_data, Y_data.reshape(len(X_data), -1),
        test_size=0.25, random_state=42
    )
    X_test, X_val, Y_test, Y_val = train_test_split(
        X_train, Y_train, test_size=0.5, random_state=42
    )

    print("\nTraining M1...")
    model = Lorenz96TransformerNet()
    model, history = train_model(
        model, X_train, Y_train, X_val, Y_val,
        epochs=200,
        batch_size=32,
        initial_lr=0.01
    )

    visualizer.plot_training_history(history)

    print("\nEvaluating model performance...")
    model.eval()
    with torch.no_grad():
        Y_pred = model(torch.FloatTensor(X_test))
    visualizer.plot_prediction_comparison(
        Y_test, Y_pred.numpy(),
        "Model Performance on Test Set"
    )

    print("\nTesting model generalization for F=24...")
    X_data_new, Y_data_new, _, t_new = generate_training_data(F=24)
    with torch.no_grad():
        Y_pred_before = model(torch.FloatTensor(X_data_new))
    visualizer.plot_prediction_comparison(
        Y_data_new.reshape(len(X_data_new), -1),
        Y_pred_before.numpy(),
        "Model Performance with F=24 (Before Fine-tuning)"
    )

    print("\nFine-tuning model for F=24...")
    X_train_new, X_test_new, Y_train_new, Y_test_new = train_test_split(
        X_data_new, Y_data_new.reshape(len(X_data_new), -1),
        test_size=0.25, random_state=42
    )
    X_test_new, X_val_new, Y_test_new, Y_val_new = train_test_split(
        X_train_new, Y_train_new, test_size=0.5, random_state=42
    )

    model_ft, history_ft = train_model(
        model,
        X_train_new, Y_train_new,
        X_val_new, Y_val_new,
        epochs=200,
        batch_size=32,
        initial_lr=0.01
    )

    with torch.no_grad():
        Y_pred_after = model_ft(torch.FloatTensor(X_test_new))
    visualizer.plot_prediction_comparison(
        Y_test_new,
        Y_pred_after.numpy(),
        "Model Performance with F=24 (After Fine-tuning)"
    )

    print("\nTesting hybrid system stability...")
    X0 = F_FORCING + np.random.randn(K)
    t_hybrid = np.arange(0, 10, 0.01)

    try:
        hybrid_solution = odeint(lorenz96_hybrid, X0, t_hybrid, args=(model_ft,))
        original_solution = odeint(
            lorenz96_full,
            np.concatenate([X0, np.random.randn(K*J)]),
            t_hybrid
        )

        metrics = analyze_stability(t_hybrid, hybrid_solution, original_solution)

        visualizer.plot_3d_comparison(
            original_solution[:, :3],
            Y_pred_after.numpy().reshape(-1, K, J)[:, :3, 0],
            hybrid_solution[:, :3],
            "3D System Comparison"
        )

    except Exception as e:
        print(f"Error in hybrid system simulation: {str(e)}")

if __name__ == "__main__":
    main()
