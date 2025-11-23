import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class CNNTrainer:
    """
    CNN model trainer for image classification
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = []
        
    def build_model(self):
        """
        CNN architecture banata hai
        Architecture:
        - 3 Convolutional blocks (Conv2D + MaxPooling)
        - Flatten layer
        - 2 Dense layers with Dropout
        - Softmax output layer
        """
        model = keras.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fully connected layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, X, y, test_size=0.2, augment=True):
        """
        Data ko train/validation split karta hai
        Optional: Data augmentation apply karta hai
        
        Args:
            X: numpy array of images (n_samples, height, width, channels)
            y: numpy array of labels (n_samples,)
            test_size: validation split ratio
            augment: whether to apply data augmentation
        """
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Data augmentation sirf training data pe
        if augment:
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.15,
                fill_mode='nearest'
            )
            datagen.fit(X_train)
            return datagen, X_train, X_val, y_train, y_val
        
        return None, X_train, X_val, y_train, y_val
    
    def train(self, X, y, class_names, epochs=20, batch_size=32, augment=True):
        """
        Model ko train karta hai
        
        Args:
            X: Images array (n_samples, 224, 224, 3)
            y: Labels array (n_samples,)
            class_names: List of class names
            epochs: Training epochs
            batch_size: Batch size
            augment: Apply data augmentation
        
        Returns:
            Dictionary with training results
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Build model
        self.build_model()
        
        print(f"🔧 Model architecture built with {self.num_classes} classes")
        print(f"📊 Total images: {len(X)}")
        
        # Prepare data
        datagen, X_train, X_val, y_train, y_val = self.prepare_data(
            X, y, augment=augment
        )
        
        print(f"✅ Data split - Train: {len(X_train)}, Validation: {len(X_val)}")
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
        
        # Train
        print("\n🚀 Training started...")
        
        if augment and datagen:
            self.history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
        
        # Evaluate
        print("\n📈 Evaluating model...")
        train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        
        # Predictions for confusion matrix
        y_pred = np.argmax(self.model.predict(X_val), axis=1)
        conf_matrix = confusion_matrix(y_val, y_pred)
        
        print(f"\n✅ Training completed!")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        
        return {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'confusion_matrix': conf_matrix,
            'history': self.history.history,
            'class_names': self.class_names
        }
    
    def plot_training_history(self):
        """Training history ko plot karta hai"""
        if self.history is None:
            print("No training history available")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, conf_matrix):
        """Confusion matrix plot karta hai"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        return fig
    
    def save_model(self, filepath='models/trained_model.h5'):
        """Model ko save karta hai"""
        if self.model:
            self.model.save(filepath)
            print(f"✅ Model saved to {filepath}")
        else:
            print("❌ No model to save")
    
    def load_model(self, filepath='models/trained_model.h5'):
        """Saved model ko load karta hai"""
        try:
            self.model = keras.models.load_model(filepath)
            print(f"✅ Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def predict(self, image):
        """
        Single image ka prediction karta hai
        
        Args:
            image: numpy array (224, 224, 3) normalized [0, 1]
        
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise Exception("Model not trained or loaded!")
        
        # Reshape for batch prediction
        image = np.expand_dims(image, axis=0)
        
        # Predict
        predictions = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # All class probabilities
        class_probabilities = {
            self.class_names[i]: float(predictions[0][i])
            for i in range(len(self.class_names))
        }
        
        return {
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': class_probabilities
        }