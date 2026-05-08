"""
Assignment 5: HPC Application for AI/ML Domain
Simple parallel model training demonstration
"""

import time
import numpy as np
from multiprocessing import Pool, cpu_count


class SimpleModel:
    """Simple linear model for demonstration"""

    def __init__(self, n_features):
        self.weights = np.random.randn(n_features)
        self.bias = 0

    def predict(self, X):
        """Make predictions"""
        return np.dot(X, self.weights) + self.bias

    def train_batch(self, X, y, lr=0.01):
        """Train on a batch"""
        predictions = self.predict(X)
        error = predictions - y

        # Update weights
        self.weights -= lr * np.dot(X.T, error) / len(X)
        self.bias -= lr * np.mean(error)

        # Calculate loss
        loss = np.mean(error ** 2)
        return loss


def generate_data(n_samples, n_features):
    """Generate synthetic data"""
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    y = np.dot(X, true_weights) + np.random.randn(n_samples) * 0.1
    return X, y


def train_sequential(X, y, n_features, epochs, batch_size, lr):
    """Sequential training"""
    model = SimpleModel(n_features)
    n_samples = len(X)
    n_batches = n_samples // batch_size

    print("  Training progress:")
    for epoch in range(epochs):
        total_loss = 0

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            X_batch = X[start:end]
            y_batch = y[start:end]

            loss = model.train_batch(X_batch, y_batch, lr)
            total_loss += loss

        avg_loss = total_loss / n_batches

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    return model, avg_loss


def train_batch_parallel(args):
    """Worker function for parallel training"""
    X_batch, y_batch, weights, bias, lr, n_features = args

    # Create local model
    model = SimpleModel(n_features)
    model.weights = weights.copy()
    model.bias = bias

    # Train
    loss = model.train_batch(X_batch, y_batch, lr)

    return model.weights, model.bias, loss


def train_parallel(X, y, n_features, epochs, batch_size, lr, n_workers):
    """Parallel training using data parallelism"""
    model = SimpleModel(n_features)
    n_samples = len(X)
    n_batches = n_samples // batch_size

    print("  Training progress:")
    for epoch in range(epochs):
        # Prepare batches
        batch_args = []
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            X_batch = X[start:end]
            y_batch = y[start:end]

            batch_args.append((
                X_batch, y_batch,
                model.weights, model.bias,
                lr, n_features
            ))

        # Train batches in parallel
        with Pool(n_workers) as pool:
            results = pool.map(train_batch_parallel, batch_args)

        # Average weights from all batches
        weights_list = [w for w, b, l in results]
        bias_list = [b for w, b, l in results]
        loss_list = [l for w, b, l in results]

        model.weights = np.mean(weights_list, axis=0)
        model.bias = np.mean(bias_list)
        avg_loss = np.mean(loss_list)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    return model, avg_loss


def evaluate(model, X, y):
    """Evaluate model"""
    predictions = model.predict(X)
    mse = np.mean((predictions - y) ** 2)
    return mse


def main():
    print("=" * 60)
    print("Assignment 5: HPC for AI/ML Domain")
    print("Parallel Model Training with Data Parallelism")
    print("=" * 60)

    # Configuration
    n_samples = 1000
    n_features = 10
    epochs = 50
    batch_size = 32
    lr = 0.01
    n_workers = min(cpu_count(), 4)

    print(f"\nConfiguration:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  CPU cores: {cpu_count()}")
    print(f"  Workers: {n_workers}")

    # Generate data
    print("\nGenerating data...")
    X, y = generate_data(n_samples, n_features)

    # Split data
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Sequential Training
    print("\n" + "-" * 60)
    print("SEQUENTIAL TRAINING")
    print("-" * 60)
    start = time.time()
    model_seq, loss_seq = train_sequential(X_train, y_train, n_features, epochs, batch_size, lr)
    time_seq = time.time() - start

    mse_seq = evaluate(model_seq, X_test, y_test)
    print(f"\nResults:")
    print(f"  Time: {time_seq:.2f}s")
    print(f"  Final Loss: {loss_seq:.6f}")
    print(f"  Test MSE: {mse_seq:.6f}")

    # Parallel Training
    print("\n" + "-" * 60)
    print("PARALLEL TRAINING")
    print("-" * 60)
    start = time.time()
    model_par, loss_par = train_parallel(X_train, y_train, n_features, epochs, batch_size, lr, n_workers)
    time_par = time.time() - start

    mse_par = evaluate(model_par, X_test, y_test)
    print(f"\nResults:")
    print(f"  Time: {time_par:.2f}s")
    print(f"  Final Loss: {loss_par:.6f}")
    print(f"  Test MSE: {mse_par:.6f}")

    # Comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    speedup = time_seq / time_par
    print(f"Speedup: {speedup:.2f}x")
    print(f"Efficiency: {speedup / n_workers * 100:.2f}%")

    print("\nHPC Techniques Used:")
    print("  • Data Parallelism - Split data across workers")
    print("  • Batch Processing - Process multiple samples together")
    print("  • Gradient Averaging - Combine results from workers")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
