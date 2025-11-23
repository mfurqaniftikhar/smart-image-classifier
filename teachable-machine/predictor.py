import numpy as np
from tensorflow import keras
import cv2
from PIL import Image
import os
import json

class Predictor:
    """
    Trained CNN model ko load karke predictions karta hai
    Ye class separate prediction service ke liye use ho sakti hai
    """
    
    def __init__(self, model_path='models/trained_model.h5', config_path='models/config.json'):
        """
        Initialize predictor with model path
        
        Args:
            model_path: Path to saved .h5 model file
            config_path: Path to config file (class names, etc.)
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.class_names = []
        self.input_shape = (224, 224, 3)
        self.is_loaded = False
    
    def load_model(self):
        """
        Saved model ko load karta hai
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found at {self.model_path}")
                return False
            
            # Load Keras model
            print(f"🔄 Loading model from {self.model_path}...")
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ Model loaded successfully!")
            
            # Load config if exists
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.class_names = config.get('class_names', [])
                    print(f"✅ Loaded {len(self.class_names)} classes: {self.class_names}")
            else:
                print(f"⚠️ Config file not found. Using default class names.")
                # Generate default class names based on output shape
                num_classes = self.model.output_shape[-1]
                self.class_names = [f"Class_{i}" for i in range(num_classes)]
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image):
        """
        Image ko model ke liye preprocess karta hai
        
        Args:
            image: PIL Image, numpy array, or file path
        
        Returns:
            Preprocessed numpy array (1, 224, 224, 3)
        """
        # Handle different input types
        if isinstance(image, str):
            # File path
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            image = Image.open(image)
        
        if isinstance(image, Image.Image):
            # PIL Image to numpy
            image = np.array(image)
        
        # Ensure RGB
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image, top_k=None):
        """
        Single image ka prediction karta hai
        
        Args:
            image: PIL Image, numpy array, or file path
            top_k: Return top-k predictions (None = all classes)
        
        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded:
            raise Exception("Model not loaded! Call load_model() first.")
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        probabilities = predictions[0]
        
        # Get predicted class
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Create results dictionary
        results = {
            'predicted_class': predicted_class,
            'predicted_index': int(predicted_idx),
            'confidence': confidence,
            'all_probabilities': {}
        }
        
        # Add all class probabilities
        for idx, class_name in enumerate(self.class_names):
            results['all_probabilities'][class_name] = float(probabilities[idx])
        
        # Sort and get top-k if specified
        if top_k:
            sorted_probs = sorted(
                results['all_probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            results['top_k_predictions'] = dict(sorted_probs)
        
        return results
    
    def predict_batch(self, images):
        """
        Multiple images ka prediction ek saath karta hai
        
        Args:
            images: List of images (PIL, numpy, or file paths)
        
        Returns:
            List of prediction dictionaries
        """
        if not self.is_loaded:
            raise Exception("Model not loaded! Call load_model() first.")
        
        results = []
        
        print(f"🔄 Processing {len(images)} images...")
        
        for idx, image in enumerate(images):
            try:
                result = self.predict(image)
                results.append(result)
                print(f"  [{idx+1}/{len(images)}] Predicted: {result['predicted_class']} ({result['confidence']:.2%})")
            except Exception as e:
                print(f"  ❌ Error processing image {idx+1}: {str(e)}")
                results.append(None)
        
        return results
    
    def predict_from_webcam(self, num_frames=1, show_preview=False):
        """
        Webcam se live prediction karta hai
        
        Args:
            num_frames: Number of frames to predict
            show_preview: Show camera preview window
        
        Returns:
            List of predictions
        """
        if not self.is_loaded:
            raise Exception("Model not loaded! Call load_model() first.")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise Exception("❌ Could not open webcam")
        
        print("📸 Webcam opened. Press 'q' to quit.")
        
        predictions = []
        frame_count = 0
        
        try:
            while frame_count < num_frames:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Failed to grab frame")
                    break
                
                # Show preview
                if show_preview:
                    cv2.imshow('Webcam Preview', frame)
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Predict
                result = self.predict(frame_rgb)
                predictions.append(result)
                
                print(f"Frame {frame_count + 1}: {result['predicted_class']} ({result['confidence']:.2%})")
                
                frame_count += 1
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        return predictions
    
    def evaluate_on_folder(self, folder_path, true_label=None):
        """
        Ek folder ke saare images ka prediction karta hai
        
        Args:
            folder_path: Path to folder containing images
            true_label: True label for accuracy calculation (optional)
        
        Returns:
            Dictionary with evaluation results
        """
        if not self.is_loaded:
            raise Exception("Model not loaded! Call load_model() first.")
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Get all image files
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]
        
        if not image_files:
            print(f"⚠️ No images found in {folder_path}")
            return None
        
        print(f"📁 Found {len(image_files)} images in {folder_path}")
        
        # Predict all images
        predictions = self.predict_batch(image_files)
        
        # Calculate statistics
        results = {
            'total_images': len(image_files),
            'predictions': predictions,
            'class_distribution': {}
        }
        
        # Count predictions per class
        for pred in predictions:
            if pred:
                class_name = pred['predicted_class']
                results['class_distribution'][class_name] = \
                    results['class_distribution'].get(class_name, 0) + 1
        
        # Calculate accuracy if true label provided
        if true_label and true_label in self.class_names:
            correct = sum(1 for p in predictions if p and p['predicted_class'] == true_label)
            results['accuracy'] = correct / len(predictions)
            results['true_label'] = true_label
            print(f"\n✅ Accuracy: {results['accuracy']:.2%}")
        
        return results
    
    def save_config(self):
        """
        Model configuration ko save karta hai (class names, etc.)
        """
        config = {
            'class_names': self.class_names,
            'input_shape': list(self.input_shape),
            'num_classes': len(self.class_names)
        }
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"✅ Config saved to {self.config_path}")
    
    def get_model_summary(self):
        """
        Model architecture ka summary return karta hai
        """
        if not self.is_loaded:
            return "Model not loaded"
        
        from io import StringIO
        import sys
        
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary = stream.getvalue()
        stream.close()
        
        return summary


# ============================================
# STANDALONE USAGE EXAMPLE
# ============================================
if __name__ == "__main__":
    """
    Predictor ko standalone script ke taur pe use kar sakte ho
    """
    
    print("=" * 50)
    print("🤖 CNN Predictor - Standalone Mode")
    print("=" * 50)
    
    # Initialize predictor
    predictor = Predictor(
        model_path='models/trained_model.h5',
        config_path='models/config.json'
    )
    
    # Load model
    if not predictor.load_model():
        print("\n❌ Failed to load model. Train a model first!")
        exit(1)
    
    print("\n" + "=" * 50)
    print("Available Operations:")
    print("1. Predict single image")
    print("2. Predict from folder")
    print("3. Predict from webcam")
    print("4. Show model summary")
    print("=" * 50)
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        # Single image prediction
        image_path = input("Enter image path: ").strip()
        
        if os.path.exists(image_path):
            print("\n🔄 Making prediction...")
            result = predictor.predict(image_path, top_k=3)
            
            print("\n" + "=" * 50)
            print("📊 PREDICTION RESULTS")
            print("=" * 50)
            print(f"🎯 Predicted Class: {result['predicted_class']}")
            print(f"💯 Confidence: {result['confidence']:.2%}")
            print("\n📈 Top 3 Predictions:")
            for class_name, prob in result.get('top_k_predictions', {}).items():
                print(f"  - {class_name}: {prob:.2%}")
        else:
            print(f"❌ File not found: {image_path}")
    
    elif choice == "2":
        # Folder prediction
        folder_path = input("Enter folder path: ").strip()
        true_label = input("Enter true label (optional, press Enter to skip): ").strip()
        
        if os.path.exists(folder_path):
            results = predictor.evaluate_on_folder(
                folder_path,
                true_label if true_label else None
            )
            
            print("\n" + "=" * 50)
            print("📊 FOLDER EVALUATION RESULTS")
            print("=" * 50)
            print(f"Total Images: {results['total_images']}")
            print("\nClass Distribution:")
            for class_name, count in results['class_distribution'].items():
                percentage = (count / results['total_images']) * 100
                print(f"  - {class_name}: {count} ({percentage:.1f}%)")
        else:
            print(f"❌ Folder not found: {folder_path}")
    
    elif choice == "3":
        # Webcam prediction
        num_frames = int(input("How many frames to capture? (default: 5): ").strip() or "5")
        predictions = predictor.predict_from_webcam(num_frames=num_frames, show_preview=True)
        
        print("\n" + "=" * 50)
        print("📸 WEBCAM PREDICTIONS SUMMARY")
        print("=" * 50)
        
        # Count most common prediction
        from collections import Counter
        pred_classes = [p['predicted_class'] for p in predictions]
        most_common = Counter(pred_classes).most_common(1)[0]
        
        print(f"Most Common Prediction: {most_common[0]} ({most_common[1]}/{len(predictions)} frames)")
    
    elif choice == "4":
        # Show model summary
        print("\n" + "=" * 50)
        print("🏗️ MODEL ARCHITECTURE")
        print("=" * 50)
        print(predictor.get_model_summary())
    
    else:
        print("❌ Invalid choice!")
    
    print("\n✅ Done!")