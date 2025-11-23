"""
Smart Model Selector - Dataset analysis aur model recommendations
"""

class ModelSelector:
    """
    Dataset analyze karke best model recommend karta hai
    User ko choice deta hai ki kaunsa model use kare
    """
    
    @staticmethod
    def analyze_dataset(total_images, num_classes):
        """
        Dataset ko analyze karke SABHI models ki predictions deta hai
        
        Args:
            total_images: Total number of images
            num_classes: Number of classes
        
        Returns:
            dict with analysis for all models
        """
        avg_per_class = total_images / num_classes if num_classes > 0 else 0
        
        analysis = {
            'total_images': total_images,
            'num_classes': num_classes,
            'avg_per_class': avg_per_class,
            'models': {}
        }
        
        # Logistic Regression Analysis
        if total_images >= 10:
            lr_accuracy = ModelSelector._estimate_accuracy('logistic_regression', total_images, num_classes)
            analysis['models']['logistic_regression'] = {
                'suitable': True,
                'estimated_accuracy': lr_accuracy,
                'confidence': ModelSelector._get_confidence(total_images, 10, 50),
                'recommendation': ModelSelector._get_recommendation('logistic_regression', total_images),
                'training_time': '30 seconds - 1 minute',
                'prediction_speed': 'Very Fast (< 0.1s per image)',
                'memory_usage': 'Low (< 100 MB)',
                'best_scenario': 'Quick prototyping, limited data',
                'emoji': '⚡',
                'color': '#F59E0B'
            }
        else:
            analysis['models']['logistic_regression'] = {
                'suitable': False,
                'reason': f'Need at least 10 images (current: {total_images})',
                'emoji': '⚡',
                'color': '#F59E0B'
            }
        
        # Random Forest Analysis
        if total_images >= 50:
            rf_accuracy = ModelSelector._estimate_accuracy('random_forest', total_images, num_classes)
            analysis['models']['random_forest'] = {
                'suitable': True,
                'estimated_accuracy': rf_accuracy,
                'confidence': ModelSelector._get_confidence(total_images, 50, 100),
                'recommendation': ModelSelector._get_recommendation('random_forest', total_images),
                'training_time': '1-3 minutes',
                'prediction_speed': 'Fast (< 0.2s per image)',
                'memory_usage': 'Medium (100-300 MB)',
                'best_scenario': 'Balanced accuracy and speed',
                'emoji': '🌲',
                'color': '#10B981'
            }
        else:
            analysis['models']['random_forest'] = {
                'suitable': False,
                'reason': f'Need at least 50 images (current: {total_images}). Add {50 - total_images} more!',
                'emoji': '🌲',
                'color': '#10B981'
            }
        
        # CNN Analysis
        if total_images >= 100:
            cnn_accuracy = ModelSelector._estimate_accuracy('cnn', total_images, num_classes)
            analysis['models']['cnn'] = {
                'suitable': True,
                'estimated_accuracy': cnn_accuracy,
                'confidence': ModelSelector._get_confidence(total_images, 100, 500),
                'recommendation': ModelSelector._get_recommendation('cnn', total_images),
                'training_time': '3-10 minutes',
                'prediction_speed': 'Medium (0.3-0.5s per image)',
                'memory_usage': 'High (500 MB - 2 GB)',
                'best_scenario': 'Maximum accuracy with sufficient data',
                'emoji': '🧠',
                'color': '#3B82F6'
            }
        else:
            analysis['models']['cnn'] = {
                'suitable': False,
                'reason': f'Need at least 100 images (current: {total_images}). Add {100 - total_images} more!',
                'emoji': '🧠',
                'color': '#3B82F6'
            }
        
        # Best model recommendation
        analysis['best_model'] = ModelSelector._get_best_model(total_images)
        
        return analysis
    
    @staticmethod
    def _estimate_accuracy(model_type, total_images, num_classes):
        """
        Expected accuracy estimate karta hai based on dataset size
        """
        # Base accuracy calculations
        if model_type == 'logistic_regression':
            if total_images < 20:
                base = 0.55
            elif total_images < 35:
                base = 0.62
            else:
                base = 0.68
            
        elif model_type == 'random_forest':
            if total_images < 70:
                base = 0.68
            elif total_images < 90:
                base = 0.75
            else:
                base = 0.80
            
        else:  # CNN
            if total_images < 150:
                base = 0.78
            elif total_images < 300:
                base = 0.85
            elif total_images < 500:
                base = 0.90
            else:
                base = 0.93
        
        # Adjust for number of classes (more classes = slightly harder)
        class_penalty = (num_classes - 2) * 0.02 if num_classes > 2 else 0
        final_accuracy = max(0.45, min(0.98, base - class_penalty))
        
        return {
            'min': max(0.40, final_accuracy - 0.05),
            'expected': final_accuracy,
            'max': min(0.98, final_accuracy + 0.08)
        }
    
    @staticmethod
    def _get_confidence(total_images, min_images, optimal_images):
        """
        Model confidence level return karta hai
        """
        if total_images < min_images:
            return 'Not Suitable'
        elif total_images < min_images + (optimal_images - min_images) * 0.3:
            return 'Low Confidence'
        elif total_images < min_images + (optimal_images - min_images) * 0.7:
            return 'Medium Confidence'
        else:
            return 'High Confidence'
    
    @staticmethod
    def _get_recommendation(model_type, total_images):
        """
        Model-specific recommendations
        """
        recommendations = {
            'logistic_regression': {
                'low': 'Suitable for quick testing. Consider adding more images for better accuracy.',
                'medium': 'Good for fast prototyping. Random Forest may give better results.',
                'high': 'Works well! But Random Forest or CNN will likely perform better.'
            },
            'random_forest': {
                'low': 'Will work, but needs more images for optimal performance.',
                'medium': 'Good balance of speed and accuracy for this dataset size.',
                'high': 'Excellent choice! Great accuracy without CNN complexity.'
            },
            'cnn': {
                'low': 'Will work, but needs significantly more images for best results.',
                'medium': 'Good choice! More images will improve accuracy further.',
                'high': 'Perfect! CNN will deliver best possible accuracy with this much data.'
            }
        }
        
        if total_images < 35:
            level = 'low'
        elif total_images < 150:
            level = 'medium'
        else:
            level = 'high'
        
        return recommendations[model_type][level]
    
    @staticmethod
    def _get_best_model(total_images):
        """
        Best model determine karta hai
        """
        if total_images < 10:
            return None
        elif total_images < 50:
            return 'logistic_regression'
        elif total_images < 100:
            return 'random_forest'
        else:
            return 'cnn'
    
    @staticmethod
    def select_model(total_images):
        """
        Backward compatibility ke liye (purani code ke liye)
        """
        if total_images < 10:
            return {
                'model_type': None,
                'reason': 'Dataset too small. Need at least 10 images total.',
                'min_images': 10
            }
        
        elif total_images <= 50:
            return {
                'model_type': 'logistic_regression',
                'reason': 'Small dataset (10-50 images). Using Logistic Regression for fast training.',
                'min_images': 10,
                'emoji': '⚡',
                'color': '#F59E0B'
            }
        
        elif total_images <= 100:
            return {
                'model_type': 'random_forest',
                'reason': 'Medium dataset (50-100 images). Using Random Forest for better accuracy.',
                'min_images': 50,
                'emoji': '🌲',
                'color': '#10B981'
            }
        
        else:
            return {
                'model_type': 'cnn',
                'reason': 'Large dataset (100+ images). Using CNN for best accuracy.',
                'min_images': 100,
                'emoji': '🧠',
                'color': '#3B82F6'
            }
    
    @staticmethod
    def get_model_info(model_type):
        """
        Model ke bare me detailed info return karta hai
        """
        info = {
            'logistic_regression': {
                'name': 'Logistic Regression',
                'description': 'Fast and simple linear classifier',
                'pros': ['Very fast training', 'Low memory usage', 'Good for small datasets'],
                'cons': ['Limited accuracy', 'Cannot learn complex patterns'],
                'training_time': '< 1 minute',
                'best_for': 'Quick testing with limited data'
            },
            'random_forest': {
                'name': 'Random Forest',
                'description': 'Ensemble of decision trees',
                'pros': ['Better accuracy than Logistic Regression', 'Handles non-linear patterns', 'Fast training'],
                'cons': ['Moderate accuracy', 'Larger model size'],
                'training_time': '1-2 minutes',
                'best_for': 'Medium-sized datasets (50-100 images)'
            },
            'cnn': {
                'name': 'Convolutional Neural Network (CNN)',
                'description': 'Deep learning model optimized for images',
                'pros': ['Highest accuracy', 'Learns complex features', 'State-of-the-art performance'],
                'cons': ['Slower training', 'Needs more data', 'Higher memory usage'],
                'training_time': '2-10 minutes',
                'best_for': 'Large datasets (100+ images)'
            }
        }
        
        return info.get(model_type, {})