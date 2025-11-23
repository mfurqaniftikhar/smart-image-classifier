import streamlit as st
from PIL import Image
import numpy as np
from cnn_trainer import CNNTrainer
from random_forest_trainer import RandomForestTrainer
from logistic_regression_trainer import LogisticRegressionTrainer
from model_selector import ModelSelector
from predictor import Predictor
from utils import (
    preprocess_image, 
    validate_image, 
    validate_dataset,
    prepare_dataset_for_training
)
import os
import json

# Cache image loading for better performance
@st.cache_data
def load_and_validate_image(file_bytes, filename):
    """Load and validate image with caching"""
    try:
        img = Image.open(file_bytes)
        return img, True, "Valid"
    except Exception as e:
        return None, False, str(e)

# Page configuration
st.set_page_config(
    page_title="Teachable Machine - Smart Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E40AF;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-box {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 20px;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #10B981;
        margin-bottom: 15px;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #F59E0B;
    }
    .error-box {
        background-color: #FEE2E2;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #EF4444;
    }
    .info-box {
        background-color: #DBEAFE;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #3B82F6;
        margin: 15px 0;
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'class_images' not in st.session_state:
    st.session_state.class_images = {}

if 'cnn_trainer' not in st.session_state:
    st.session_state.cnn_trainer = CNNTrainer()

if 'rf_trainer' not in st.session_state:
    st.session_state.rf_trainer = RandomForestTrainer()

if 'lr_trainer' not in st.session_state:
    st.session_state.lr_trainer = LogisticRegressionTrainer()

if 'active_trainer' not in st.session_state:
    st.session_state.active_trainer = None

if 'selected_model_type' not in st.session_state:
    st.session_state.selected_model_type = None

if 'predictor' not in st.session_state:
    st.session_state.predictor = Predictor()

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

if 'training_results' not in st.session_state:
    st.session_state.training_results = None

if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False

# Header
st.markdown('<div class="main-header">🤖 Teachable Machine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Smart AI that automatically picks the best model for your data!</div>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Choose a step:",
    ["1️⃣ Upload Images", "2️⃣ Train Model", "3️⃣ Test Predictions"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Sidebar stats
if st.session_state.class_images:
    total_images = sum(len(imgs) for imgs in st.session_state.class_images.values())
    st.sidebar.markdown("### 📊 Current Dataset")
    st.sidebar.metric("Classes", len(st.session_state.class_images))
    st.sidebar.metric("Total Images", total_images)
    
    # Show recommended model
    model_recommendation = ModelSelector.select_model(total_images)
    if model_recommendation['model_type']:
        st.sidebar.markdown("### 🎯 Recommended Model")
        model_emoji = model_recommendation.get('emoji', '🤖')
        st.sidebar.info(f"{model_emoji} **{model_recommendation['model_type'].replace('_', ' ').title()}**")
    
    if st.session_state.model_trained:
        st.sidebar.success("✅ Model Trained")
        if st.session_state.selected_model_type:
            st.sidebar.write(f"Active: **{st.session_state.selected_model_type.replace('_', ' ').title()}**")
    else:
        st.sidebar.warning("⚠️ Model Not Trained")

st.sidebar.markdown("---")
st.sidebar.info("""
**How it works:**
1. Upload images (any amount)
2. AI picks best model automatically
3. Train and test predictions!

**Models:**
- ⚡ Logistic Regression (10-50 images)
- 🌲 Random Forest (50-100 images)
- 🧠 CNN (100+ images)
""")

# Advanced options in sidebar
with st.sidebar.expander("⚙️ Advanced Options"):
    if st.button("🗑️ Clear All Data", type="secondary"):
        st.session_state.class_images = {}
        st.session_state.model_trained = False
        st.session_state.training_results = None
        st.session_state.selected_model_type = None
        st.session_state.active_trainer = None
        st.success("✅ All data cleared!")
        st.rerun()
    
    if st.button("💾 Export Dataset Info"):
        if st.session_state.class_images:
            dataset_info = {
                'num_classes': len(st.session_state.class_images),
                'classes': {k: len(v) for k, v in st.session_state.class_images.items()},
                'total_images': sum(len(v) for v in st.session_state.class_images.values())
            }
            st.json(dataset_info)
        else:
            st.warning("No dataset to export")

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data/uploaded', exist_ok=True)

# ============================================
# PAGE 1: UPLOAD IMAGES
# ============================================
if page == "1️⃣ Upload Images":
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.header("📸 Step 1: Create Classes & Upload Images")
    st.write("Upload images - AI will automatically choose the best model!")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input for new class
    col1, col2 = st.columns([3, 1])
    with col1:
        new_class = st.text_input(
            "Enter class name (e.g., 'Cat', 'Dog', 'Car')",
            key="new_class_input",
            placeholder="Type class name here..."
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add Class", type="primary", use_container_width=True):
            if new_class and new_class.strip():
                new_class = new_class.strip()
                if new_class not in st.session_state.class_images:
                    st.session_state.class_images[new_class] = []
                    st.success(f"✅ Class '{new_class}' added!")
                    st.rerun()
                else:
                    st.warning("⚠️ Class already exists!")
            else:
                st.error("❌ Please enter a class name")
    
    st.markdown("---")
    
    # Display existing classes
    if not st.session_state.class_images:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 👋 Welcome! Let's get started
        
        **Quick Guide:**
        1. Enter a class name above (e.g., "Cat", "Dog", "Car")
        2. Click "➕ Add Class"
        3. Upload images for each class
        4. AI automatically picks the best model for your data size!
        
        **Model Selection:**
        - 📊 **10-50 images**: Logistic Regression (fast)
        - 🌲 **50-100 images**: Random Forest (balanced)
        - 🧠 **100+ images**: CNN (most accurate)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.subheader(f"📚 Classes ({len(st.session_state.class_images)})")
        
        # Create tabs for each class
        class_tabs = st.tabs([f"🏷️ {name}" for name in st.session_state.class_images.keys()])
        
        for tab_idx, (class_name, images) in enumerate(st.session_state.class_images.items()):
            with class_tabs[tab_idx]:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"### {class_name}")
                    st.write(f"**Images: {len(images)}**")
                    
                    # Progress indicator
                    if len(images) < 5:
                        st.warning(f"⚠️ Need {5 - len(images)} more images (minimum 5 required)")
                    elif len(images) < 10:
                        st.info(f"💡 Add {10 - len(images)} more for better results")
                    else:
                        st.success(f"✅ Great! {len(images)} images uploaded")
                
                with col2:
                    # Delete class button
                    if st.button(f"🗑️ Delete Class", key=f"del_{class_name}", type="secondary"):
                        if st.session_state.get(f'confirm_delete_{class_name}', False):
                            del st.session_state.class_images[class_name]
                            st.session_state[f'confirm_delete_{class_name}'] = False
                            st.rerun()
                        else:
                            st.session_state[f'confirm_delete_{class_name}'] = True
                            st.warning("Click again to confirm deletion")
                
                st.markdown("---")
                
                # Upload images for this class
                uploaded_files = st.file_uploader(
                    f"Upload images for {class_name}",
                    type=['jpg', 'jpeg', 'png'],
                    accept_multiple_files=True,
                    key=f"upload_{class_name}",
                    help="Select multiple images (Ctrl/Cmd + Click)"
                )
                
                # Process uploaded files
                if uploaded_files:
                    upload_key = f"processed_{class_name}"
                    if upload_key not in st.session_state:
                        st.session_state[upload_key] = set()
                    
                    new_images = []
                    for uploaded_file in uploaded_files:
                        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                        
                        if file_id not in st.session_state[upload_key]:
                            is_valid, message = validate_image(uploaded_file)
                            if is_valid:
                                uploaded_file.seek(0)
                                img = Image.open(uploaded_file)
                                new_images.append(img)
                                st.session_state[upload_key].add(file_id)
                    
                    if new_images:
                        st.session_state.class_images[class_name].extend(new_images)
                        st.success(f"✅ Added {len(new_images)} new images")
                        st.rerun()
                
                # Display uploaded images
                if images:
                    st.markdown("#### 🖼️ Uploaded Images")
                    
                    show_images = st.checkbox(f"Show images ({len(images)})", value=False, key=f"show_{class_name}")
                    
                    if show_images:
                        display_limit = st.slider(
                            "Images to display", 
                            min_value=4, 
                            max_value=min(len(images), 50), 
                            value=min(len(images), 20),
                            key=f"limit_{class_name}"
                        )
                        
                        cols_per_row = 4
                        for i in range(0, min(display_limit, len(images)), cols_per_row):
                            img_cols = st.columns(cols_per_row)
                            for j, img_col in enumerate(img_cols):
                                if i + j < len(images) and i + j < display_limit:
                                    with img_col:
                                        display_img = images[i + j].copy()
                                        display_img.thumbnail((200, 200))
                                        st.image(display_img, use_column_width=True)
                                        if st.button(f"🗑️", key=f"del_img_{class_name}_{i+j}", 
                                                   help="Delete this image"):
                                            del st.session_state.class_images[class_name][i + j]
                                            st.rerun()
                        
                        if len(images) > display_limit:
                            st.info(f"Showing {display_limit} of {len(images)} images")
                else:
                    st.info("👆 Upload your first image using the button above")
        
        st.markdown("---")
        
        # Dataset summary with model recommendation
        total_images = sum(len(imgs) for imgs in st.session_state.class_images.values())
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.write(f"### 📊 Dataset Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Classes", len(st.session_state.class_images))
        with col2:
            st.metric("Total Images", total_images)
        with col3:
            avg_images = total_images / len(st.session_state.class_images) if st.session_state.class_images else 0
            st.metric("Avg per Class", f"{avg_images:.1f}")
        
        # Model recommendation
        model_rec = ModelSelector.select_model(total_images)
        if model_rec['model_type']:
            st.markdown("---")
            st.markdown("### 🎯 AI Recommendation")
            model_emoji = model_rec.get('emoji', '🤖')
            model_color = model_rec.get('color', '#3B82F6')
            
            st.markdown(f"""
            <div style="background-color: {model_color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {model_color};">
                <h3 style="color: {model_color}; margin: 0;">
                    {model_emoji} {model_rec['model_type'].replace('_', ' ').title()}
                </h3>
                <p style="margin: 10px 0 0 0; font-size: 1.1rem;">
                    {model_rec['reason']}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Validation
        is_valid, msg = validate_dataset(st.session_state.class_images)
        st.markdown("---")
        if is_valid:
            st.success(f"✅ {msg} - Ready to train!")
        else:
            st.warning(f"⚠️ {msg}")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PAGE 2: TRAIN MODEL (UPDATED WITH USER CHOICE)
# ============================================
# ============================================
# PAGE 2: TRAIN MODEL (FIXED VERSION)
# ============================================
elif page == "2️⃣ Train Model":
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.header("🚀 Step 2: Train AI Model")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Check if dataset is ready
    is_valid, msg = validate_dataset(st.session_state.class_images)
    
    if not is_valid:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning(f"⚠️ {msg}")
        st.write("Please go back to Step 1 and upload more images.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        total_images = sum(len(imgs) for imgs in st.session_state.class_images.values())
        num_classes = len(st.session_state.class_images)
        
        # 🆕 ANALYZE DATASET FOR ALL MODELS
        st.subheader("📊 Dataset Analysis & Model Predictions")
        
        analysis = ModelSelector.analyze_dataset(total_images, num_classes)
        
        # Display dataset stats - FIXED
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Images", int(analysis['total_images']))
        with col2:
            st.metric("Classes", int(analysis['num_classes']))
        with col3:
            st.metric("Avg per Class", f"{float(analysis['avg_per_class']):.1f}")
        
        st.markdown("---")
        
        # 🆕 DISPLAY ALL MODELS WITH PREDICTIONS
        st.subheader("🎯 Model Comparison & Predictions")
        st.write("See how each model will perform on your dataset:")
        
        # Create columns for each model
        model_cols = st.columns(3)
        
        available_models = []
        
        for idx, model_type in enumerate(['logistic_regression', 'random_forest', 'cnn']):
            model_data = analysis['models'][model_type]
            model_info = ModelSelector.get_model_info(model_type)
            
            with model_cols[idx]:
                # Model card design
                if model_data['suitable']:
                    # Suitable model - show predictions
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {model_data['color']}ee 0%, {model_data['color']}99 100%);
                        padding: 20px;
                        border-radius: 15px;
                        color: white;
                        min-height: 400px;
                    ">
                        <h3 style="margin: 0; color: white; text-align: center;">
                            {model_data['emoji']} {model_info['name']}
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<div style='margin-top: -380px; padding: 20px;'>", unsafe_allow_html=True)
                    
                    # Accuracy prediction
                    acc = model_data['estimated_accuracy']
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin: 10px 0;">
                        <p style="margin:0; color: white; font-size: 0.9rem;">Expected Accuracy:</p>
                        <p style="margin:5px 0 0 0; color: white; font-size: 1.8rem; font-weight: bold;">
                            {float(acc['expected']):.1%}
                        </p>
                        <p style="margin:5px 0 0 0; color: rgba(255,255,255,0.8); font-size: 0.85rem;">
                            Range: {float(acc['min']):.1%} - {float(acc['max']):.1%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence badge
                    confidence_colors = {
                        'High Confidence': '#10B981',
                        'Medium Confidence': '#F59E0B',
                        'Low Confidence': '#EF4444'
                    }
                    conf_color = confidence_colors.get(model_data['confidence'], '#6B7280')
                    
                    st.markdown(f"""
                    <div style="background: {conf_color}; padding: 8px; border-radius: 5px; text-align: center; margin: 10px 0;">
                        <strong style="color: white; font-size: 0.9rem;">{model_data['confidence']}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Performance metrics
                    st.markdown(f"""
                    <div style="color: white; font-size: 0.85rem; margin: 15px 0;">
                        <p style="margin: 5px 0;">⏱️ Training: {model_data['training_time']}</p>
                        <p style="margin: 5px 0;">⚡ Speed: {model_data['prediction_speed']}</p>
                        <p style="margin: 5px 0;">💾 Memory: {model_data['memory_usage']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Recommendation
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.15); padding: 10px; border-radius: 8px; margin: 10px 0;">
                        <p style="margin:0; color: white; font-size: 0.8rem; line-height: 1.4;">
                            💡 {model_data['recommendation']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    available_models.append(model_type)
                    
                else:
                    # Not suitable model - show reason
                    st.markdown(f"""
                    <div style="
                        background: #F3F4F6;
                        padding: 20px;
                        border-radius: 15px;
                        border: 2px dashed #D1D5DB;
                        min-height: 400px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        align-items: center;
                        text-align: center;
                    ">
                        <div style="font-size: 3rem; margin-bottom: 10px; opacity: 0.3;">
                            {model_data['emoji']}
                        </div>
                        <h3 style="color: #6B7280; margin: 10px 0;">
                            {model_info['name']}
                        </h3>
                        <p style="color: #9CA3AF; font-size: 0.9rem; margin: 10px 0;">
                            Not Available
                        </p>
                        <p style="color: #6B7280; font-size: 0.85rem; padding: 10px; background: white; border-radius: 8px; margin: 10px 0;">
                            {model_data['reason']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 🆕 USER MODEL SELECTION
        if available_models:
            st.subheader("🎯 Choose Your Model")
            
            # Show recommended model
            best_model = analysis['best_model']
            if best_model:
                best_info = ModelSelector.get_model_info(best_model)
                best_data = analysis['models'][best_model]
                st.info(f"💡 **AI Recommendation:** {best_data['emoji']} **{best_info['name']}** - Expected accuracy: {float(best_data['estimated_accuracy']['expected']):.1%}")
            
            # Model selection radio buttons
            model_options = []
            model_labels = []
            
            for model_type in available_models:
                info = ModelSelector.get_model_info(model_type)
                data = analysis['models'][model_type]
                label = f"{data['emoji']} {info['name']} (Accuracy: ~{float(data['estimated_accuracy']['expected']):.1%})"
                model_options.append(model_type)
                model_labels.append(label)
            
            # Default selection (recommended model)
            default_idx = model_options.index(best_model) if best_model in model_options else 0
            
            selected_model = st.radio(
                "Select the model you want to train:",
                options=model_options,
                format_func=lambda x: model_labels[model_options.index(x)],
                index=default_idx,
                horizontal=False
            )
            
            # Display selected model details
            selected_info = ModelSelector.get_model_info(selected_model)
            selected_data = analysis['models'][selected_model]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {selected_data['color']}20 0%, {selected_data['color']}10 100%);
                        padding: 20px; border-radius: 10px; border-left: 5px solid {selected_data['color']}; margin: 20px 0;">
                <h3 style="color: {selected_data['color']}; margin: 0;">
                    {selected_data['emoji']} Selected: {selected_info['name']}
                </h3>
                <p style="margin: 10px 0; font-size: 1.05rem;">
                    <strong>Expected Performance:</strong> {float(selected_data['estimated_accuracy']['expected']):.1%} accuracy
                </p>
                <p style="margin: 5px 0; color: #6B7280;">
                    {selected_data['recommendation']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Training configuration based on selected model
            st.subheader("⚙️ Training Configuration")
            
            if selected_model == 'cnn':
                col1, col2, col3 = st.columns(3)
                with col1:
                    epochs = st.slider("Epochs", 5, 50, 20, 5)
                with col2:
                    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
                with col3:
                    augment = st.checkbox("Data Augmentation", value=True)
            elif selected_model == 'random_forest':
                n_estimators = st.slider("Number of Trees", 50, 200, 100, 50,
                                        help="More trees = better accuracy but slower")
            else:  # logistic_regression
                max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100,
                                   help="Maximum optimization iterations")
            
            # Dataset split info - FIXED
            st.markdown("### 📊 Training Split")
            train_size = int(total_images * 0.8)
            val_size = int(total_images - train_size)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Classes", int(num_classes))
            with col2:
                st.metric("Total Images", int(total_images))
            with col3:
                st.metric("Training Set", int(train_size))
            with col4:
                st.metric("Validation Set", int(val_size))
            
            st.markdown("---")
            
            # Train button
            if st.button("🎯 Start Training", type="primary", use_container_width=True, 
                        disabled=st.session_state.training_in_progress):
                st.session_state.training_in_progress = True
                st.session_state.selected_model_type = selected_model
                
                with st.spinner("🔄 Preparing dataset..."):
                    X, y, class_names = prepare_dataset_for_training(
                        st.session_state.class_images
                    )
                
                st.write("---")
                st.subheader("📈 Training Progress")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text(f"🚀 Initializing {selected_info['name']}...")
                    progress_bar.progress(10)
                    
                    # Select trainer based on model type
                    if selected_model == 'cnn':
                        trainer = st.session_state.cnn_trainer
                        results = trainer.train(X, y, class_names, epochs=epochs, 
                                              batch_size=batch_size, augment=augment)
                        trainer.save_model('models/trained_model.h5')
                    elif selected_model == 'random_forest':
                        trainer = st.session_state.rf_trainer
                        results = trainer.train(X, y, class_names, n_estimators=n_estimators)
                        trainer.save_model('models/random_forest_model.pkl')
                    else:  # logistic_regression
                        trainer = st.session_state.lr_trainer
                        results = trainer.train(X, y, class_names, max_iter=max_iter)
                        trainer.save_model('models/logistic_regression_model.pkl')
                    
                    st.session_state.active_trainer = trainer
                    progress_bar.progress(100)
                    
                    # Save results
                    st.session_state.training_results = results
                    st.session_state.model_trained = True
                    st.session_state.training_in_progress = False
                    
                    status_text.text("✅ Training completed successfully!")
                    st.balloons()
                    
                    # Display results - FIXED
                    st.markdown("---")
                    st.subheader("📊 Training Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Training Accuracy", f"{float(results['train_accuracy']):.2%}")
                    with col2:
                        st.metric("Validation Accuracy", f"{float(results['val_accuracy']):.2%}")
                    
                    # Compare with prediction - FIXED
                    expected_acc = float(selected_data['estimated_accuracy']['expected'])
                    actual_acc = float(results['val_accuracy'])
                    
                    if actual_acc >= expected_acc - 0.05:
                        st.success(f"🎉 Great! Actual accuracy ({actual_acc:.1%}) matches or exceeds prediction ({expected_acc:.1%})!")
                    elif actual_acc >= expected_acc - 0.10:
                        st.info(f"👍 Good! Actual accuracy ({actual_acc:.1%}) is close to prediction ({expected_acc:.1%})")
                    else:
                        st.warning(f"💡 Actual accuracy ({actual_acc:.1%}) is lower than predicted ({expected_acc:.1%}). Try adding more diverse images!")
                    
                    # Confusion matrix
                    st.markdown("---")
                    st.subheader("🔍 Confusion Matrix")
                    fig_cm = trainer.plot_confusion_matrix(results['confusion_matrix'])
                    st.pyplot(fig_cm)
                    
                except Exception as e:
                    st.session_state.training_in_progress = False
                    st.error(f"❌ Training failed: {str(e)}")
                    import traceback
                    with st.expander("🔍 View Error Details"):
                        st.code(traceback.format_exc())
            
            # Show previous results if available - FIXED
            if st.session_state.model_trained and st.session_state.training_results:
                st.markdown("---")
                st.success(f"✅ {st.session_state.selected_model_type.replace('_', ' ').title()} is trained!")
                
                with st.expander("📊 View Training Results"):
                    results = st.session_state.training_results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Train Accuracy", f"{float(results['train_accuracy']):.2%}")
                    with col2:
                        st.metric("Validation Accuracy", f"{float(results['val_accuracy']):.2%}")
        
        else:
            st.error("❌ No suitable models available for this dataset size. Please add more images!")

# ============================================
# PAGE 3: TEST PREDICTIONS
# ============================================
elif page == "3️⃣ Test Predictions":
    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.header("🔮 Step 3: Test Predictions")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained or not st.session_state.active_trainer:
        st.warning("⚠️ Please train the model first (Step 2)")
    else:
        # Show active model
        st.success(f"✅ Active Model: **{st.session_state.selected_model_type.replace('_', ' ').title()}**")
        
        # Prediction mode selection
        st.markdown("### 🎯 Choose Prediction Mode")
        pred_mode = st.radio("", ["📷 Upload Image", "📸 Use Webcam", "📁 Batch Predict"],
                           horizontal=True, label_visibility="collapsed")
        
        st.markdown("---")
        
        trainer = st.session_state.active_trainer
        
        # Upload Image Mode
        if pred_mode == "📷 Upload Image":
            st.subheader("📷 Upload an Image")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file:
                is_valid, message = validate_image(uploaded_file)
                
                if is_valid:
                    image = Image.open(uploaded_file)
                    with col1:
                        st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    with st.spinner("🔄 Making prediction..."):
                        processed_img = preprocess_image(image)
                        result = trainer.predict(processed_img)
                    
                    with col2:
                        st.markdown("### 🎯 Prediction Results")
                        st.markdown(f"""
                        <div class="success-box">
                        <h2 style="color: #10B981; margin:0; text-align: center;">
                            {result['predicted_class']}
                        </h2>
                        <p style="font-size: 1.5rem; margin:10px 0; text-align: center;">
                            Confidence: <strong>{result['confidence']:.2%}</strong>
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if result['confidence'] > 0.9:
                            st.success("🎯 Very confident!")
                        elif result['confidence'] > 0.7:
                            st.info("👍 Good confidence")
                        else:
                            st.warning("⚠️ Low confidence")
                        
                        st.markdown("---")
                        st.write("**All Class Probabilities:**")
                        sorted_probs = sorted(result['probabilities'].items(), 
                                            key=lambda x: x[1], reverse=True)
                        for class_name, prob in sorted_probs:
                            st.progress(prob, text=f"{class_name}: {prob:.2%}")
                else:
                    st.error(f"❌ {message}")
        
        # Webcam Mode
        elif pred_mode == "📸 Use Webcam":
            st.subheader("📸 Capture from Webcam")
            camera_image = st.camera_input("Take a picture", label_visibility="collapsed")
            
            if camera_image:
                image = Image.open(camera_image)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Captured Image", use_column_width=True)
                
                with st.spinner("🔄 Making prediction..."):
                    processed_img = preprocess_image(image)
                    result = trainer.predict(processed_img)
                
                with col2:
                    st.markdown("### 🎯 Results")
                    st.markdown(f"""
                    <div class="success-box">
                    <h2 style="color: #10B981; text-align: center;">{result['predicted_class']}</h2>
                    <p style="font-size: 1.5rem; text-align: center;">
                        Confidence: <strong>{result['confidence']:.2%}</strong>
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.write("**All Class Probabilities:**")
                    sorted_probs = sorted(result['probabilities'].items(), 
                                        key=lambda x: x[1], reverse=True)
                    for class_name, prob in sorted_probs:
                        st.progress(prob, text=f"{class_name}: {prob:.2%}")

        # Batch Prediction Mode
        else:
            st.subheader("📁 Batch Prediction")
            st.write("Upload multiple images for batch prediction")
            
            uploaded_files = st.file_uploader(
                "Choose multiple images...",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                st.write(f"📊 Processing {len(uploaded_files)} images...")
                
                batch_results = []
                with st.spinner("Processing images..."):
                    for uploaded_file in uploaded_files:
                        is_valid, message = validate_image(uploaded_file)
                        
                        if is_valid:
                            uploaded_file.seek(0)
                            image = Image.open(uploaded_file)
                            processed_img = preprocess_image(image)
                            result = trainer.predict(processed_img)
                            
                            batch_results.append({
                                'filename': uploaded_file.name,
                                'image': image,
                                'prediction': result
                            })
                
                st.success(f"✅ Processed {len(batch_results)} images!")
                
                # Summary
                st.markdown("### 📊 Batch Results Summary")
                class_counts = {}
                for res in batch_results:
                    pred_class = res['prediction']['predicted_class']
                    class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.bar_chart(class_counts)
                with col2:
                    for class_name, count in class_counts.items():
                        percentage = (count / len(batch_results)) * 100
                        st.metric(class_name, f"{count} ({percentage:.1f}%)")
                
                st.markdown("---")
                st.markdown("### 📋 Individual Results")
                
                # Display results in grid
                cols_per_row = 3
                for i in range(0, len(batch_results), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(batch_results):
                            result = batch_results[i + j]
                            with col:
                                st.image(result['image'], use_column_width=True)
                                pred = result['prediction']
                                confidence_color = "#10B981" if pred['confidence'] > 0.8 else "#F59E0B"
                                
                                st.markdown(f"""
                                <div style="background-color: #F3F4F6; padding: 10px; 
                                           border-radius: 5px; margin-top: 5px;">
                                    <p style="margin: 0; font-size: 0.8rem; color: #6B7280;">
                                        {result['filename']}
                                    </p>
                                    <p style="margin: 5px 0 0 0; font-size: 1.1rem; 
                                              font-weight: bold; color: {confidence_color};">
                                        {pred['predicted_class']} ({pred['confidence']:.1%})
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Export results
                st.markdown("---")
                if st.button("📥 Export Results as JSON"):
                    export_data = [{
                        'filename': res['filename'],
                        'predicted_class': res['prediction']['predicted_class'],
                        'confidence': res['prediction']['confidence'],
                        'all_probabilities': res['prediction']['probabilities']
                    } for res in batch_results]
                    
                    json_str = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="💾 Download JSON",
                        data=json_str,
                        file_name="batch_predictions.json",
                        mime="application/json"
                    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; padding: 20px;">
    <p>Built with ❤️ using Streamlit & scikit-learn/TensorFlow</p>
    <p style="font-size: 0.9rem;">Smart AI: Logistic Regression | Random Forest | CNN</p>
    <p style="font-size: 0.8rem; margin-top: 10px;">
        <a href="https://github.com" target="_blank" style="color: #3B82F6; text-decoration: none;">📚 Documentation</a> | 
        <a href="https://github.com" target="_blank" style="color: #3B82F6; text-decoration: none;">⭐ GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)
