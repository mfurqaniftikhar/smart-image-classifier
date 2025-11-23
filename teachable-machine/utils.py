import numpy as np
from PIL import Image
import io

def preprocess_image(image, target_size=(224, 224)):
    """
    Image ko CNN ke liye preprocess karta hai
    
    Args:
        image: PIL Image object
        target_size: tuple (height, width) - MUST match model input shape
    
    Returns:
        numpy array (height, width, 3) normalized [0, 1]
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to target size (224, 224)
    image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    return img_array


def validate_image(uploaded_file, max_size_mb=10):
    """
    Uploaded image ko validate karta hai
    
    Args:
        uploaded_file: Streamlit uploaded file object
        max_size_mb: Maximum file size in MB
    
    Returns:
        tuple: (is_valid, message)
    """
    try:
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File too large ({file_size_mb:.1f}MB). Max size: {max_size_mb}MB"
        
        # Try to open image
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        
        # Check if image is valid
        image.verify()
        
        # Check dimensions (should be at least 32x32)
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        if image.size[0] < 32 or image.size[1] < 32:
            return False, "Image too small. Minimum size: 32x32 pixels"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Invalid image: {str(e)}"


def validate_dataset(class_images, min_images_per_class=5, min_classes=2):
    """
    Dataset ko validate karta hai training se pehle
    
    Args:
        class_images: dict {class_name: [images]}
        min_images_per_class: minimum images required per class
        min_classes: minimum number of classes required
    
    Returns:
        tuple: (is_valid, message)
    """
    # Check minimum classes
    if len(class_images) < min_classes:
        return False, f"Need at least {min_classes} classes. Currently: {len(class_images)}"
    
    # Check images per class
    for class_name, images in class_images.items():
        if len(images) < min_images_per_class:
            return False, f"Class '{class_name}' has only {len(images)} images. Need at least {min_images_per_class}"
    
    total_images = sum(len(images) for images in class_images.values())
    return True, f"Dataset valid: {len(class_images)} classes, {total_images} images"


def prepare_dataset_for_training(class_images, target_size=(224, 224)):
    """
    Dataset ko training ke liye prepare karta hai
    
    Args:
        class_images: dict {class_name: [PIL Images]}
        target_size: tuple (height, width) for preprocessing
    
    Returns:
        tuple: (X, y, class_names)
            X: numpy array (n_samples, height, width, 3)
            y: numpy array (n_samples,) - class indices
            class_names: list of class names
    """
    X = []
    y = []
    class_names = list(class_images.keys())
    
    print(f"📦 Preparing dataset for {len(class_names)} classes...")
    
    for class_idx, (class_name, images) in enumerate(class_images.items()):
        print(f"  Processing {class_name}: {len(images)} images")
        
        for img in images:
            # Preprocess image with correct target size
            processed_img = preprocess_image(img, target_size=target_size)
            X.append(processed_img)
            y.append(class_idx)
    
    # Convert to numpy arrays
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int32')
    
    print(f"✅ Dataset prepared: X shape = {X.shape}, y shape = {y.shape}")
    
    return X, y, class_names


def get_image_stats(image):
    """
    Image ki basic stats return karta hai
    
    Args:
        image: PIL Image object
    
    Returns:
        dict with image statistics
    """
    img_array = np.array(image)
    
    return {
        'size': image.size,
        'mode': image.mode,
        'format': image.format,
        'shape': img_array.shape,
        'dtype': img_array.dtype,
        'min_value': img_array.min(),
        'max_value': img_array.max(),
        'mean_value': img_array.mean()
    }