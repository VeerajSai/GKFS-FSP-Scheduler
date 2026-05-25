"""
Sequence Models for FSP Scheduling
===================================

LSTM and BiGRU-CNN hybrid models for time-series prediction.
These models predict FSP error to enable intelligent switching.

Maintainer: Project Team
Date: January 2026
"""

import numpy as np
from typing import Tuple, Dict, Optional

# TensorFlow is optional - models will only work if installed
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        Input, LSTM, GRU, Bidirectional, Dense, Dropout,
        BatchNormalization, Conv1D, GlobalMaxPooling1D,
        Concatenate, Layer
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    Model = None  # Type placeholder


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM/GRU models.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target array (n_samples,)
    sequence_length : int
        Number of timesteps to look back

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        X_seq: (n_samples - sequence_length, sequence_length, n_features)
        y_seq: (n_samples - sequence_length,)
    """
    X_seq = []
    y_seq = []

    for i in range(sequence_length, len(X)):
        X_seq.append(X[i - sequence_length:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)


def build_lstm_model(
    input_shape: Tuple[int, int],
    units: list = [128, 64],
    dropout_rate: float = 0.3,
    recurrent_dropout: float = 0.2,
    learning_rate: float = 0.001
) -> Model:
    """
    Build LSTM model for time-series regression.

    Parameters:
    -----------
    input_shape : Tuple[int, int]
        (sequence_length, n_features)
    units : list
        Number of units in each LSTM layer
    dropout_rate : float
        Dropout rate for regularization
    recurrent_dropout : float
        Dropout for recurrent connections
    learning_rate : float
        Learning rate for Adam optimizer

    Returns:
    --------
    Model : Compiled Keras model
    """
    model = Sequential([
        LSTM(
            units[0],
            return_sequences=True if len(units) > 1 else False,
            input_shape=input_shape,
            recurrent_dropout=recurrent_dropout
        ),
        BatchNormalization(),
        Dropout(dropout_rate),
    ])

    # Add additional LSTM layers
    for i, n_units in enumerate(units[1:], start=1):
        return_seq = i < len(units) - 1
        model.add(LSTM(n_units, return_sequences=return_seq, recurrent_dropout=recurrent_dropout))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate * 0.7))  # Reduce dropout in deeper layers

    # Output layer
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    return model


def build_bigru_cnn_model(
    input_shape: Tuple[int, int],
    gru_units: list = [64, 32],
    cnn_filters: int = 64,
    cnn_kernel_size: int = 3,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001
) -> Model:
    """
    Build BiGRU-CNN hybrid model.

    Combines:
    - Bidirectional GRU for capturing temporal patterns in both directions
    - 1D CNN for local pattern extraction

    Parameters:
    -----------
    input_shape : Tuple[int, int]
        (sequence_length, n_features)
    gru_units : list
        Number of units in each GRU layer
    cnn_filters : int
        Number of CNN filters
    cnn_kernel_size : int
        CNN kernel size
    dropout_rate : float
        Dropout rate
    learning_rate : float
        Learning rate

    Returns:
    --------
    Model : Compiled Keras model
    """
    inputs = Input(shape=input_shape)

    # Bidirectional GRU branch
    x_gru = inputs
    for i, units in enumerate(gru_units):
        return_seq = i < len(gru_units) - 1
        x_gru = Bidirectional(
            GRU(units, return_sequences=return_seq or i == 0)
        )(x_gru)
        x_gru = BatchNormalization()(x_gru)
        x_gru = Dropout(dropout_rate)(x_gru)

    # If we still have sequences, apply global pooling
    if len(x_gru.shape) == 3:
        x_gru = GlobalMaxPooling1D()(x_gru)

    # CNN branch
    x_cnn = Conv1D(
        filters=cnn_filters,
        kernel_size=cnn_kernel_size,
        activation='relu',
        padding='same'
    )(inputs)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = Conv1D(
        filters=cnn_filters // 2,
        kernel_size=cnn_kernel_size,
        activation='relu',
        padding='same'
    )(x_cnn)
    x_cnn = GlobalMaxPooling1D()(x_cnn)
    x_cnn = Dropout(dropout_rate)(x_cnn)

    # Combine branches
    combined = Concatenate()([x_gru, x_cnn])

    # Dense layers
    x = Dense(64, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.5)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    return model


def get_callbacks(
    model_path: str,
    patience: int = 15,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    monitor: str = 'val_mae'
) -> list:
    """
    Get standard callbacks for training.

    Parameters:
    -----------
    model_path : str
        Path to save best model
    patience : int
        Early stopping patience
    reduce_lr_patience : int
        Reduce LR patience
    reduce_lr_factor : float
        Factor to reduce LR by
    monitor : str
        Metric to monitor

    Returns:
    --------
    list : List of Keras callbacks
    """
    return [
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor=monitor,
            save_best_only=True,
            verbose=0
        )
    ]


def train_sequence_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_path: str,
    epochs: int = 150,
    batch_size: int = 64,
    patience: int = 15
) -> dict:
    """
    Train a sequence model with standard configuration.

    Returns:
    --------
    dict : Training history
    """
    callbacks = get_callbacks(model_path, patience=patience)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history.history
