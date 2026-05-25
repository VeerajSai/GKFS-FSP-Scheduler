"""
Model Builder Utilities
========================

Comprehensive model building functions for:
- Deep Learning models (ANN, LSTM, GRU, Temporal CNN, Custom)
- Ensemble models (Averaging, Weighted, Stacking, Hybrids)

Maintainer: Project Team
Date: January 2026
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks


def get_time_index(df, time_col=None):
    """Get a numeric time index from a dataframe."""
    if time_col and time_col in df.columns:
        series = df[time_col]
    elif 'timestamp' in df.columns:
        series = df['timestamp']
    elif 'block' in df.columns:
        series = df['block']
    else:
        return df.index.to_numpy(dtype=float)

    if np.issubdtype(series.dtype, np.datetime64):
        return pd.to_datetime(series).view('int64') / 3.6e12  # hours

    try:
        return series.astype(float).to_numpy()
    except Exception:
        return np.arange(len(df), dtype=float)


def build_harmonic_features(time_index, periods, order):
    """Create harmonic (sin/cos) features for given periods and order."""
    time_index = np.asarray(time_index, dtype=float)
    features = []
    for period in periods:
        for k in range(1, order + 1):
            omega = 2 * np.pi * k / max(period, 1e-6)
            features.append(np.sin(omega * time_index))
            features.append(np.cos(omega * time_index))
    if not features:
        return np.empty((len(time_index), 0))
    return np.column_stack(features)


def build_harmonic_regression_features(X_scaled, df, config):
    """Build design matrix for harmonic regression."""
    periods = config.get('periods', [24, 96])
    order = config.get('order', 2)
    time_col = config.get('time_col')
    time_index = get_time_index(df, time_col=time_col)
    harmonic_features = build_harmonic_features(time_index, periods, order)

    if config.get('use_original_features', True):
        if harmonic_features.size == 0:
            return X_scaled
        return np.column_stack([X_scaled, harmonic_features])

    return harmonic_features


def compute_fuzzy_entropy_feature(X, r=0.2):
    """Lightweight fuzzy-entropy proxy per sample."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    z = (X - mean) / std
    weights = np.exp(-np.abs(z) / max(r, 1e-6))
    probs = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
    return entropy.reshape(-1, 1)


def apply_lightweight_decomposition(X, window=9):
    """Lightweight trend/residual decomposition across time axis."""
    X = np.asarray(X, dtype=float)
    if window < 3:
        return X
    kernel = np.ones(window, dtype=float) / window
    low_pass = np.zeros_like(X)
    for i in range(X.shape[1]):
        low_pass[:, i] = np.convolve(X[:, i], kernel, mode='same')
    high_pass = X - low_pass
    return np.column_stack([X, low_pass, high_pass])


def prepare_ceemdan_vmd_features(X_scaled, config):
    """Prepare CEEMDAN+VMD-inspired features with safe fallback."""
    window = int(config.get('decomp_window', 9))
    return apply_lightweight_decomposition(X_scaled, window=window)


def prepare_ivmd_features(X_scaled, config):
    """Prepare improved VMD features with safe fallback."""
    window = int(config.get('decomp_window', 7))
    return apply_lightweight_decomposition(X_scaled, window=window)


def prepare_ivmd_fe_features(X_scaled, config):
    """Prepare IVMD + Fuzzy Entropy features."""
    X_ivmd = prepare_ivmd_features(X_scaled, config)
    fe = compute_fuzzy_entropy_feature(X_ivmd, r=config.get('fuzzy_r', 0.2))
    return np.column_stack([X_ivmd, fe])


def adaptive_huber_loss(delta_floor=1e-3):
    """Adaptive Huber loss using batch error statistics."""
    import tensorflow as tf

    def loss_fn(y_true, y_pred):
        error = y_true - y_pred
        delta = tf.stop_gradient(tf.reduce_mean(tf.abs(error))) + delta_floor
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return tf.reduce_mean(0.5 * tf.square(quadratic) + delta * linear)

    return loss_fn


def build_ann_model(input_dim, config):
    """Build Artificial Neural Network (Dense layers)."""
    model = models.Sequential(name='ANN')

    # Input layer
    model.add(layers.Input(shape=(input_dim,)))

    # Hidden layers
    for i, units in enumerate(config['layers']):
        model.add(layers.Dense(units, activation=config.get('activation', 'relu'), name=f'dense_{i+1}'))
        if config.get('dropout', 0) > 0:
            model.add(layers.Dropout(config['dropout'], name=f'dropout_{i+1}'))

    # Output layer
    model.add(layers.Dense(1, activation='linear', name='output'))

    # Compile
    optimizer_name = config.get('optimizer', 'adam')
    lr = config.get('learning_rate', 0.001)

    if optimizer_name == 'adam':
        opt = optimizers.Adam(learning_rate=lr)
    elif optimizer_name == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=lr)
    else:
        opt = optimizers.SGD(learning_rate=lr)

    model.compile(optimizer=opt, loss='mse', metrics=['mae'])

    return model


def build_fcnn_model(input_dim, config):
    """Build Fully Connected Neural Network with batch normalization."""
    model = models.Sequential(name='FCNN')

    # Input layer
    model.add(layers.Input(shape=(input_dim,)))

    # Hidden layers
    for i in range(config['num_layers']):
        model.add(layers.Dense(config['layer_size'], activation='relu', name=f'dense_{i+1}'))

        if config.get('batch_norm', False):
            model.add(layers.BatchNormalization(name=f'batchnorm_{i+1}'))

        if config.get('dropout', 0) > 0:
            model.add(layers.Dropout(config['dropout'], name=f'dropout_{i+1}'))

    # Output layer
    model.add(layers.Dense(1, activation='linear', name='output'))

    # Compile
    lr = config.get('learning_rate', 0.001)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse', metrics=['mae'])

    return model


def build_lstm_model(input_dim, config):
    """Build LSTM model for sequence prediction."""
    model = models.Sequential(name='LSTM')

    timesteps = config.get('timesteps', 3)
    input_shape = (timesteps, input_dim // timesteps)

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # LSTM layers
    units_list = config['units']
    bidirectional = config.get('bidirectional', False)

    for i, units in enumerate(units_list):
        return_sequences = (i < len(units_list) - 1)

        lstm_layer = layers.LSTM(
            units,
            return_sequences=return_sequences,
            dropout=config.get('dropout', 0),
            recurrent_dropout=config.get('recurrent_dropout', 0),
            name=f'lstm_{i+1}'
        )

        if bidirectional:
            lstm_layer = layers.Bidirectional(lstm_layer, name=f'bidirectional_lstm_{i+1}')

        model.add(lstm_layer)

    # Dense output
    model.add(layers.Dense(64, activation='relu', name='dense_1'))
    model.add(layers.Dropout(config.get('dropout', 0), name='dropout_final'))
    model.add(layers.Dense(1, activation='linear', name='output'))

    # Compile
    lr = config.get('learning_rate', 0.001)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse', metrics=['mae'])

    return model


def build_gru_model(input_dim, config):
    """Build GRU model for sequence prediction."""
    model = models.Sequential(name='GRU')

    timesteps = config.get('timesteps', 3)
    input_shape = (timesteps, input_dim // timesteps)

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # GRU layers
    units_list = config['units']
    bidirectional = config.get('bidirectional', False)

    for i, units in enumerate(units_list):
        return_sequences = (i < len(units_list) - 1)

        gru_layer = layers.GRU(
            units,
            return_sequences=return_sequences,
            dropout=config.get('dropout', 0),
            recurrent_dropout=config.get('recurrent_dropout', 0),
            name=f'gru_{i+1}'
        )

        if bidirectional:
            gru_layer = layers.Bidirectional(gru_layer, name=f'bidirectional_gru_{i+1}')

        model.add(gru_layer)

    # Dense output
    model.add(layers.Dense(64, activation='relu', name='dense_1'))
    model.add(layers.Dropout(config.get('dropout', 0), name='dropout_final'))
    model.add(layers.Dense(1, activation='linear', name='output'))

    # Compile
    lr = config.get('learning_rate', 0.001)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse', metrics=['mae'])

    return model


def build_temporal_cnn_model(input_dim, config):
    """Build Temporal Convolutional Neural Network."""
    model = models.Sequential(name='Temporal_CNN')

    timesteps = config.get('timesteps', 5)
    input_shape = (timesteps, input_dim // timesteps)

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # Convolutional layers
    for i in range(config['num_conv_layers']):
        model.add(layers.Conv1D(
            filters=config['num_filters'],
            kernel_size=config['kernel_size'],
            activation='relu',
            padding='causal',
            name=f'conv1d_{i+1}'
        ))

        if config.get('pool_size', 1) > 1:
            model.add(layers.MaxPooling1D(pool_size=config['pool_size'], name=f'maxpool_{i+1}'))

        model.add(layers.Dropout(config.get('dropout', 0), name=f'dropout_conv_{i+1}'))

    # Flatten
    model.add(layers.Flatten(name='flatten'))

    # Dense layers
    for i, units in enumerate(config['dense_units']):
        model.add(layers.Dense(units, activation='relu', name=f'dense_{i+1}'))
        model.add(layers.Dropout(config.get('dropout', 0), name=f'dropout_dense_{i+1}'))

    # Output
    model.add(layers.Dense(1, activation='linear', name='output'))

    # Compile
    lr = config.get('learning_rate', 0.001)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse', metrics=['mae'])

    return model


def build_custom_architecture(input_dim, config):
    """Build custom neural network architecture based on user configuration."""
    arch_type = config['architecture_type']
    layers_config = config['layers_config']
    dropout = config.get('dropout', 0.3)
    lr = config.get('learning_rate', 0.001)

    if arch_type == "Dense Only":
        model = models.Sequential(name='Custom_Dense')
        model.add(layers.Input(shape=(input_dim,)))

        layer_sizes = [int(x.strip()) for x in layers_config.split(',')]
        for i, size in enumerate(layer_sizes):
            model.add(layers.Dense(size, activation='relu', name=f'dense_{i+1}'))
            model.add(layers.Dropout(dropout, name=f'dropout_{i+1}'))

        model.add(layers.Dense(1, activation='linear', name='output'))

    elif arch_type == "CNN + Dense":
        timesteps = config.get('timesteps', 5)
        input_shape = (timesteps, input_dim // timesteps)

        model = models.Sequential(name='Custom_CNN_Dense')
        model.add(layers.Input(shape=input_shape))

        # CNN layers
        cnn_filters = [int(x.strip()) for x in layers_config['cnn'].split(',')]
        for i, filters in enumerate(cnn_filters):
            model.add(layers.Conv1D(filters, kernel_size=3, activation='relu', padding='causal', name=f'conv_{i+1}'))
            model.add(layers.MaxPooling1D(pool_size=2, name=f'pool_{i+1}'))

        model.add(layers.Flatten(name='flatten'))

        # Dense layers
        dense_sizes = [int(x.strip()) for x in layers_config['dense'].split(',')]
        for i, size in enumerate(dense_sizes):
            model.add(layers.Dense(size, activation='relu', name=f'dense_{i+1}'))
            model.add(layers.Dropout(dropout, name=f'dropout_{i+1}'))

        model.add(layers.Dense(1, activation='linear', name='output'))

    elif arch_type in ["LSTM + Dense", "GRU + Dense"]:
        timesteps = config.get('timesteps', 5)
        input_shape = (timesteps, input_dim // timesteps)

        rnn_type = "LSTM" if "LSTM" in arch_type else "GRU"
        model = models.Sequential(name=f'Custom_{rnn_type}_Dense')
        model.add(layers.Input(shape=input_shape))

        # RNN layers
        rnn_units = [int(x.strip()) for x in layers_config['rnn'].split(',')]
        for i, units in enumerate(rnn_units):
            return_seq = (i < len(rnn_units) - 1)
            if rnn_type == "LSTM":
                model.add(layers.LSTM(units, return_sequences=return_seq, dropout=dropout, name=f'lstm_{i+1}'))
            else:
                model.add(layers.GRU(units, return_sequences=return_seq, dropout=dropout, name=f'gru_{i+1}'))

        # Dense layers
        dense_sizes = [int(x.strip()) for x in layers_config['dense'].split(',')]
        for i, size in enumerate(dense_sizes):
            model.add(layers.Dense(size, activation='relu', name=f'dense_{i+1}'))
            model.add(layers.Dropout(dropout, name=f'dropout_{i+1}'))

        model.add(layers.Dense(1, activation='linear', name='output'))

    else:  # CNN + LSTM + Dense
        timesteps = config.get('timesteps', 5)
        input_shape = (timesteps, input_dim // timesteps)

        model = models.Sequential(name='Custom_CNN_LSTM_Dense')
        model.add(layers.Input(shape=input_shape))

        # CNN layers
        cnn_filters = [int(x.strip()) for x in layers_config['cnn'].split(',')]
        for i, filters in enumerate(cnn_filters):
            model.add(layers.Conv1D(filters, kernel_size=3, activation='relu', padding='causal', name=f'conv_{i+1}'))

        # LSTM layers
        lstm_units = [int(x.strip()) for x in layers_config['lstm'].split(',')]
        for i, units in enumerate(lstm_units):
            return_seq = (i < len(lstm_units) - 1)
            model.add(layers.LSTM(units, return_sequences=return_seq, dropout=dropout, name=f'lstm_{i+1}'))

        # Dense layers
        dense_sizes = [int(x.strip()) for x in layers_config['dense'].split(',')]
        for i, size in enumerate(dense_sizes):
            model.add(layers.Dense(size, activation='relu', name=f'dense_{i+1}'))
            model.add(layers.Dropout(dropout, name=f'dropout_{i+1}'))

        model.add(layers.Dense(1, activation='linear', name='output'))

    # Compile
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse', metrics=['mae'])

    return model


def build_cnn_bilstm_model(input_dim, config):
    """Build CNN + BiLSTM model for sequence prediction."""
    timesteps = config.get('timesteps', 6)
    input_shape = (timesteps, input_dim // timesteps)

    inputs = layers.Input(shape=input_shape, name='input')
    x = layers.Conv1D(
        filters=config.get('num_filters', 64),
        kernel_size=config.get('kernel_size', 3),
        activation='relu',
        padding='same',
        name='conv1d'
    )(inputs)
    x = layers.Dropout(config.get('dropout', 0.2), name='conv_dropout')(x)

    x = layers.Bidirectional(
        layers.LSTM(
            config.get('lstm_units', 64),
            return_sequences=False,
            dropout=config.get('dropout', 0.2)
        ),
        name='bilstm'
    )(x)

    x = layers.Dense(config.get('dense_units', 64), activation='relu', name='dense_1')(x)
    x = layers.Dropout(config.get('dropout', 0.2), name='dense_dropout')(x)
    outputs = layers.Dense(1, activation='linear', name='output')(x)

    model = models.Model(inputs, outputs, name='CEEMDAN_VMD_CNN_BiLSTM')
    lr = config.get('learning_rate', 0.001)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse', metrics=['mae'])

    return model


def build_informer_model(input_dim, config):
    """Build Informer-like model with adaptive loss."""
    timesteps = config.get('timesteps', 6)
    input_shape = (timesteps, input_dim // timesteps)

    inputs = layers.Input(shape=input_shape, name='input')
    x = layers.Conv1D(
        filters=config.get('d_model', 64),
        kernel_size=config.get('kernel_size', 3),
        padding='same',
        activation='relu',
        name='token_embedding'
    )(inputs)

    attention = layers.MultiHeadAttention(
        num_heads=config.get('num_heads', 4),
        key_dim=config.get('d_model', 64),
        dropout=config.get('attention_dropout', 0.1),
        name='mha'
    )(x, x)
    x = layers.Add(name='attn_add')([x, attention])
    x = layers.LayerNormalization(name='attn_norm')(x)

    ffn = layers.Dense(config.get('ff_dim', 128), activation='relu', name='ffn_1')(x)
    ffn = layers.Dense(config.get('d_model', 64), activation='relu', name='ffn_2')(ffn)
    x = layers.Add(name='ffn_add')([x, ffn])
    x = layers.LayerNormalization(name='ffn_norm')(x)

    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dropout(config.get('dropout', 0.2), name='final_dropout')(x)
    outputs = layers.Dense(1, activation='linear', name='output')(x)

    model = models.Model(inputs, outputs, name='IVMD_FE_Ad_Informer')
    lr = config.get('learning_rate', 0.001)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss=adaptive_huber_loss(), metrics=['mae'])

    return model


def reshape_for_rnn(X, timesteps):
    """Reshape data for RNN input (samples, timesteps, features)."""
    n_samples = X.shape[0]
    n_features = X.shape[1]

    # Calculate features per timestep
    features_per_timestep = n_features // timesteps

    # Trim features if not evenly divisible
    features_to_use = features_per_timestep * timesteps
    X_trimmed = X[:, :features_to_use]

    # Reshape to (samples, timesteps, features_per_timestep)
    X_reshaped = X_trimmed.reshape(n_samples, timesteps, features_per_timestep)

    return X_reshaped


class StreamlitProgressCallback(callbacks.Callback):
    """Custom callback to update Streamlit progress during training."""

    def __init__(self, progress_container, status_text, model_name, total_epochs):
        super().__init__()
        self.progress_container = progress_container
        self.status_text = status_text
        self.model_name = model_name
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.status_text.text(f"Training {self.model_name}... (Epoch {self.current_epoch}/{self.total_epochs})")
        self.progress_container.progress(self.current_epoch / self.total_epochs)

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss', 0)
        val_mae = logs.get('val_mae', 0)
        self.status_text.text(
            f"Training {self.model_name}... (Epoch {self.current_epoch}/{self.total_epochs}) "
            f"- Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}"
        )


def get_callbacks(patience=15, use_dynamic_lr=True, progress_callback=None):
    """Get training callbacks."""
    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=0
        )
    ]

    if use_dynamic_lr:
        callback_list.append(
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=0
            )
        )

    if progress_callback is not None:
        callback_list.append(progress_callback)

    return callback_list
