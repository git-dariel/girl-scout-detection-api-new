import tensorflow as tf
import os
import numpy as np
from pathlib import Path
import gc

class ModelLoader:
    _model = None
    _interpreter = None
    _input_details = None
    _output_details = None
    _tflite_model_path = None
    _base_model_path = None
    
    @classmethod
    def _init_paths(cls):
        # Get the absolute path to the models directory
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(current_dir, 'models')
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        cls._tflite_model_path = os.path.join(models_dir, 'model_optimized.tflite')
        cls._base_model_path = os.path.join(models_dir, 'keras_model.h5')
    
    @classmethod
    def get_model(cls):
        if cls._interpreter is None:
            # Initialize paths if not done yet
            if cls._tflite_model_path is None:
                cls._init_paths()
            
            # Force garbage collection before loading model
            gc.collect()
            
            try:
                # Check if optimized TFLite model exists
                if os.path.exists(cls._tflite_model_path):
                    print(f"Loading TFLite model from: {cls._tflite_model_path}")
                    # Read model in chunks to reduce memory usage
                    chunk_size = 1024 * 1024  # 1MB chunks
                    tflite_model = bytearray()
                    with open(cls._tflite_model_path, 'rb') as f:
                        while True:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            tflite_model.extend(chunk)
                else:
                    # Load and optimize the model
                    if not os.path.exists(cls._base_model_path):
                        raise FileNotFoundError(
                            f"Model files not found. Expected at:\n"
                            f"- TFLite model: {cls._tflite_model_path}\n"
                            f"- Base model: {cls._base_model_path}\n"
                            "Please ensure model files are present in the models directory."
                        )
                    
                    print(f"Loading base model from: {cls._base_model_path}")
                    # Load model with memory growth enabled
                    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
                    cls._model = tf.keras.models.load_model(cls._base_model_path)
                    
                    # Convert model to TF Lite format with optimizations
                    converter = tf.lite.TFLiteConverter.from_keras_model(cls._model)
                    
                    # Enable optimizations for reduced memory usage
                    converter.optimizations = [
                        tf.lite.Optimize.DEFAULT,
                        tf.lite.Optimize.EXPERIMENTAL_SPARSITY
                    ]
                    converter.target_spec.supported_types = [tf.float16]
                    converter.target_spec.supported_ops = [
                        tf.lite.OpsSet.TFLITE_BUILTINS,
                        tf.lite.OpsSet.SELECT_TF_OPS
                    ]
                    
                    print("Converting model to TFLite format...")
                    tflite_model = converter.convert()
                    
                    # Save the optimized model
                    print(f"Saving optimized model to: {cls._tflite_model_path}")
                    os.makedirs(os.path.dirname(cls._tflite_model_path), exist_ok=True)
                    with open(cls._tflite_model_path, 'wb') as f:
                        f.write(tflite_model)
                    
                    # Clear keras model from memory
                    del cls._model
                    gc.collect()
                
                # Create TF Lite interpreter with optimizations and reduced memory usage
                cls._interpreter = tf.lite.Interpreter(
                    model_content=bytes(tflite_model),
                    num_threads=2,  # Reduce thread count to save memory
                    experimental_preserve_all_tensors=False
                )
                
                # Clear model data from memory
                del tflite_model
                gc.collect()
                
                # Pre-allocate tensors
                cls._interpreter.allocate_tensors()
                
                # Cache input and output details
                cls._input_details = cls._interpreter.get_input_details()
                cls._output_details = cls._interpreter.get_output_details()
                
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {str(e)}")
        
        return cls._interpreter, cls._input_details, cls._output_details
