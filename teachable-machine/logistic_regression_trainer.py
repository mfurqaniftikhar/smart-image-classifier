"""
Logistic Regression Trainer for Small Datasets
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class LogisticRegressionTrainer:
    """
    Logistic Regression model trainer for small datasets
    """
    
    def __init__(self):
        self.model = None
        self.class_names = []
        self.training_results = None
        
    def train(self, X, y, class_names, test_size=0.2, max_iter=1000):
        """
        Logistic Regression model ko train karta hai
        
        Args:
            X: Images array (n_samples, height, width, channels)
            y: Labels array (n_samples,)
            class_names: List of class names
            test_size: Validation split ratio
            max_iter: Maximum iterations for optimization
            
        Returns:
            Dictionary with training results
        """
        self.class_names = class_names
        
        print(f"⚡ Training Logistic Regression...")
        print(f"📊 Total images: {len(X)}")
        
        # Flatten images to 1D vectors
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
        
        # Train-test split
        X_train, X_val, y_train, y_val = train_test_split(
            X_flat, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"✅ Data split - Train: {len(X_train)}, Validation: {len(X_val)}")
        
        # Train model
        self.model = LogisticRegression(
            max_iter=max_iter,
            multi_class='multinomial',
            solver='lbfgs',
            random_state=42,
            verbose=0
        )
        
        print("\n🚀 Training Logistic Regression (fast training)...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_val, y_val_pred)
        
        print(f"\n✅ Training completed!")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        
        self.training_results = {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'confusion_matrix': conf_matrix,
            'class_names': self.class_names
        }
        
        return self.training_results
    
    def predict(self, image):
        """
        Single image ka prediction karta hai
        
        Args:
            image: numpy array (height, width, channels) normalized [0, 1]
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise Exception("Model not trained!")
        
        # Flatten image
        img_flat = image.flatten().reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(img_flat)[0]
        probabilities = self.model.predict_proba(img_flat)[0]
        
        # Class probabilities
        class_probabilities = {
            self.class_names[i]: float(probabilities[i])
            for i in range(len(self.class_names))
        }
        
        return {
            'predicted_class': self.class_names[prediction],
            'confidence': float(probabilities[prediction]),
            'probabilities': class_probabilities
        }
    
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
        
        ax.set_title('Confusion Matrix - Logistic Regression', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        return fig
    
    def save_model(self, filepath='models/logistic_regression_model.pkl'):
        """Model ko save karta hai"""
        if self.model:
            model_data = {
                'model': self.model,
                'class_names': self.class_names
            }
            joblib.dump(model_data, filepath)
            print(f"✅ Model saved to {filepath}")
        else:
            print("❌ No model to save")
    
    def load_model(self, filepath='models/logistic_regression_model.pkl'):
        """Saved model ko load karta hai"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.class_names = model_data['class_names']
            print(f"✅ Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False