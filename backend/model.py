@@ .. @@
 import tensorflow as tf
 from tensorflow import keras
 from tensorflow.keras import layers
 import numpy as np
 import matplotlib.pyplot as plt
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import classification_report, confusion_matrix
 import seaborn as sns
+import json
 from pathlib import Path
 import logging
 
 # Configure logging
 logging.basicConfig(level=logging.INFO)
 logger = logging.getLogger(__name__)
@@ .. @@
     """
     Save trained model with metadata and labels
     
     Args:
         model: Trained Keras model
         model_name: Name for the saved model
         labels: List of class labels
         metrics: Dictionary of training metrics
         save_dir: Directory to save model files
     """
     save_path = Path(save_dir)
     save_path.mkdir(parents=True, exist_ok=True)
     
     # Save model
     model_file = save_path / f"{model_name}.h5"
     model.save(model_file)
     logger.info(f"Model saved to {model_file}")
     
    # Save labels as both JSON and plain text for compatibility
    labels_json_file = save_path / f"{model_name}.labels.json"
    labels_txt_file = save_path / f"{model_name}.labels.txt"
    
    # JSON format with metadata
    labels_data = {
        "labels": labels,
        "num_classes": len(labels),
        "model_name": model_name,
        "created_at": str(datetime.now()),
        "input_shape": model.input_shape[1:] if model.input_shape else None
    }
    
    with open(labels_json_file, 'w') as f:
        json.dump(labels_data, f, indent=2)
    logger.info(f"Labels JSON saved to {labels_json_file}")
    
    # Plain text format for easy loading
    with open(labels_txt_file, 'w') as f:
        json.dump(labels, f)  # Save as JSON array for consistency
    logger.info(f"Labels TXT saved to {labels_txt_file}")
+    
     # Save training metrics
     metrics_file = save_path / f"{model_name}_metrics.json"
     metrics_data = {
         "model_name": model_name,
         "training_date": str(datetime.now()),
         "metrics": metrics,
         "labels": labels
     }
     
     with open(metrics_file, 'w') as f:
         json.dump(metrics_data, f, indent=2)
     logger.info(f"Metrics saved to {metrics_file}")

def train_ensemble_models(data_dir: str, model_type: str = "face"):
    """
    Train ensemble of models (VGG16, ResNet50, EfficientNetB0)
    
    Args:
        data_dir: Directory containing training data
        model_type: Type of models to train ('face' or 'xray')
    """
    logger.info(f"Training ensemble models for {model_type}")
    
    # Model architectures to train
    architectures = {
        "vgg16": build_face_model if model_type == "face" else build_xray_model,
        "resnet50": build_face_model if model_type == "face" else build_xray_model,
        "efficientnet_b0": build_face_model if model_type == "face" else build_xray_model
    }
    
    # Data loading
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20 if model_type == "face" else 10,
        width_shift_range=0.2 if model_type == "face" else 0.1,
        height_shift_range=0.2 if model_type == "face" else 0.1,
        horizontal_flip=True if model_type == "face" else False,
        validation_split=0.2
    )
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    labels = list(train_generator.class_indices.keys())
    save_dir = f"models/{model_type}"
    
    # Train each architecture
    for arch_name, model_builder in architectures.items():
        logger.info(f"Training {arch_name} for {model_type}")
        
        try:
            # Build model
            model = model_builder(num_classes=train_generator.num_classes)
            
            # Training
            history = model.fit(
                train_generator,
                epochs=50,
                validation_data=validation_generator,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                ]
            )
            
            # Evaluate
            val_loss, val_accuracy = model.evaluate(validation_generator)
            
            # Save model with metadata
            metrics = {
                "architecture": arch_name,
                "val_accuracy": float(val_accuracy),
                "val_loss": float(val_loss),
                "epochs_trained": len(history.history['loss']),
                "num_classes": train_generator.num_classes
            }
            
            model_name = f"{model_type}_model_{arch_name}"
            save_model_with_metadata(model, model_name, labels, metrics, save_dir)
            
            logger.info(f"✅ {arch_name} training completed - Accuracy: {val_accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to train {arch_name}: {str(e)}")
            continue
    
    logger.info(f"Ensemble training completed for {model_type}")
+def export_model_for_serving(model_path: str, export_format: str = "onnx"):
+    """
+    Export trained model for optimized serving
+    
+    Args:
+        model_path: Path to the .h5 model file
+        export_format: Export format ('onnx', 'tflite', or 'savedmodel')
+    """
+    model_path = Path(model_path)
+    if not model_path.exists():
+        raise FileNotFoundError(f"Model file not found: {model_path}")
+    
+    # Load model
+    model = keras.models.load_model(model_path)
+    logger.info(f"Loaded model from {model_path}")
+    
+    export_path = model_path.parent / f"{model_path.stem}.{export_format}"
+    
+    if export_format.lower() == "onnx":
+        try:
+            import tf2onnx
+            import onnx
+            
+            # Convert to ONNX
+            onnx_model, _ = tf2onnx.convert.from_keras(model)
+            onnx.save(onnx_model, str(export_path))
+            logger.info(f"Model exported to ONNX: {export_path}")
+            
+        except ImportError:
+            logger.error("tf2onnx not installed. Install with: pip install tf2onnx")
+            
+    elif export_format.lower() == "tflite":
+        # Convert to TensorFlow Lite
+        converter = tf.lite.TFLiteConverter.from_keras_model(model)
+        converter.optimizations = [tf.lite.Optimize.DEFAULT]
+        tflite_model = converter.convert()
+        
+        with open(export_path, 'wb') as f:
+            f.write(tflite_model)
+        logger.info(f"Model exported to TFLite: {export_path}")
+        
+    elif export_format.lower() == "savedmodel":
+        # Export as SavedModel
+        tf.saved_model.save(model, str(export_path))
+        logger.info(f"Model exported to SavedModel: {export_path}")
+        
+    else:
+        raise ValueError(f"Unsupported export format: {export_format}")

 def train_face_model(data_dir: str, model_name: str = "face_pcos_model"):
     """
     Train facial PCOS classification model
     
     Args:
         data_dir: Directory containing training data
         model_name: Name for the saved model
     """
     logger.info(f"Training face model: {model_name}")
     
     # Data loading and preprocessing
     datagen = ImageDataGenerator(
         rescale=1./255,
         rotation_range=20,
         width_shift_range=0.2,
         height_shift_range=0.2,
         horizontal_flip=True,
         validation_split=0.2
     )
     
     train_generator = datagen.flow_from_directory(
         data_dir,
         target_size=(224, 224),
         batch_size=32,
         class_mode='categorical',
         subset='training'
     )
     
     validation_generator = datagen.flow_from_directory(
         data_dir,
         target_size=(224, 224),
         batch_size=32,
         class_mode='categorical',
         subset='validation'
     )
     
     # Build model
     model = build_face_model(num_classes=train_generator.num_classes)
     
     # Training
     history = model.fit(
         train_generator,
         epochs=50,
         validation_data=validation_generator,
         callbacks=[
             keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
             keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
         ]
     )
     
     # Evaluate
     val_loss, val_accuracy = model.evaluate(validation_generator)
     
     # Get class labels
     labels = list(train_generator.class_indices.keys())
     
     # Save model with metadata
     metrics = {
         "val_accuracy": float(val_accuracy),
         "val_loss": float(val_loss),
         "epochs_trained": len(history.history['loss']),
         "num_classes": train_generator.num_classes
     }
     
     save_model_with_metadata(model, model_name, labels, metrics)
     
     return model, history

 def train_xray_model(data_dir: str, model_name: str = "xray_pcos_model"):
     """
     Train X-ray PCOS classification model
     
     Args:
         data_dir: Directory containing training data
         model_name: Name for the saved model
     """
     logger.info(f"Training X-ray model: {model_name}")
     
     # Similar implementation to face model but with X-ray specific preprocessing
     datagen = ImageDataGenerator(
         rescale=1./255,
         rotation_range=10,  # Less rotation for medical images
         width_shift_range=0.1,
         height_shift_range=0.1,
         zoom_range=0.1,
         validation_split=0.2
     )
     
     train_generator = datagen.flow_from_directory(
         data_dir,
         target_size=(224, 224),
         batch_size=32,
         class_mode='categorical',
         subset='training'
     )
     
     validation_generator = datagen.flow_from_directory(
         data_dir,
         target_size=(224, 224),
         batch_size=32,
         class_mode='categorical',
         subset='validation'
     )
     
     # Build model
     model = build_xray_model(num_classes=train_generator.num_classes)
     
     # Training
     history = model.fit(
         train_generator,
         epochs=50,
         validation_data=validation_generator,
         callbacks=[
             keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
             keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
         ]
     )
     
     # Evaluate
     val_loss, val_accuracy = model.evaluate(validation_generator)
     
     # Get class labels
     labels = list(train_generator.class_indices.keys())
     
     # Save model with metadata
     metrics = {
         "val_accuracy": float(val_accuracy),
         "val_loss": float(val_loss),
         "epochs_trained": len(history.history['loss']),
         "num_classes": train_generator.num_classes
     }
     
     save_model_with_metadata(model, model_name, labels, metrics)
     
     return model, history

 if __name__ == "__main__":
     """
     Training script entry point
     
     Usage:
         python model.py --type face --data_dir /path/to/face/data
         python model.py --type xray --data_dir /path/to/xray/data
        python model.py --ensemble --type face --data_dir /path/to/face/data
         python model.py --export /path/to/model.h5 --format onnx
     """
     import argparse
     
     parser = argparse.ArgumentParser(description="PCOS Model Training and Export")
     parser.add_argument("--type", choices=["face", "xray"], help="Model type to train")
     parser.add_argument("--data_dir", help="Directory containing training data")
     parser.add_argument("--model_name", default="pcos_model", help="Name for saved model")
    parser.add_argument("--ensemble", action="store_true", help="Train ensemble of models")
     parser.add_argument("--export", help="Path to model file to export")
     parser.add_argument("--format", choices=["onnx", "tflite", "savedmodel"], 
                        default="onnx", help="Export format")
     
     args = parser.parse_args()
     
     if args.export:
         # Export existing model
         export_model_for_serving(args.export, args.format)
         
     elif args.type and args.data_dir:
        if args.ensemble:
            # Train ensemble of models
            train_ensemble_models(args.data_dir, args.type)
        else:
            # Train single model
            if args.type == "face":
                model, history = train_face_model(args.data_dir, args.model_name)
            elif args.type == "xray":
                model, history = train_xray_model(args.data_dir, args.model_name)
             
            print(f"\n✅ Training completed for {args.type} model: {args.model_name}")
            print(f"📁 Model files saved in models/ directory")
            print(f"📊 Check {args.model_name}_metrics.json for detailed results")
         
         # Optional: Export trained model
         export_choice = input("\nExport model for serving? (y/n): ").lower()
         if export_choice == 'y':
             format_choice = input("Export format (onnx/tflite/savedmodel): ").lower()
             if format_choice in ["onnx", "tflite", "savedmodel"]:
                if args.ensemble:
                    print("Ensemble models saved. Export individual models manually.")
                else:
                    model_path = f"models/{args.model_name}.h5"
                 export_model_for_serving(model_path, format_choice)
                 
     else:
         parser.print_help()