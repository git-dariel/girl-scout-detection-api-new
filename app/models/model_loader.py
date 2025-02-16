import tensorflow as tf
import os
import numpy as np
from pathlib import Path

class ModelLoader:
    _model = None
    _interpreter = None
    _input_details = None
    _output_details = None
    _tflite_model_path = os.path.join('models', 'model_optimized.tflite')
    
    @classmethod
    def get_model(cls):
        if cls._interpreter is None:
            # Enable TF memory growth to avoid taking all GPU memory
            physical_devices = tf.config.list_physical_devices('GPU')
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                    # Set memory limit to avoid OOM
                    tf.config.set_logical_device_configuration(
                        device,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
                    )
            except:
                pass  # No GPU available
                
            # Set TF to enable GPU memory allocation optimization
            tf.config.optimizer.set_jit(True)  # Enable XLA optimization
            
            # Check if optimized TFLite model exists
            if os.path.exists(cls._tflite_model_path):
                # Load pre-optimized TFLite model
                with open(cls._tflite_model_path, 'rb') as f:
                    tflite_model = f.read()
            else:
                # Load and optimize the model
                model_path = os.path.join('models', 'keras_model.h5')
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found at {model_path}")
                
                cls._model = tf.keras.models.load_model(model_path)
                
                # Convert model to TF Lite format with optimizations
                converter = tf.lite.TFLiteConverter.from_keras_model(cls._model)
                
                # Enable quantization and optimizations
                converter.optimizations = [
                    tf.lite.Optimize.DEFAULT,
                    tf.lite.Optimize.EXPERIMENTAL_SPARSITY
                ]
                converter.target_spec.supported_types = [tf.float16]
                
                # Enable additional optimizations
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS
                ]
                
                # Enable experimental optimizations
                converter.experimental_new_converter = True
                converter.experimental_new_quantizer = True
                
                # Convert to TFLite format
                tflite_model = converter.convert()
                
                # Save the optimized model
                os.makedirs(os.path.dirname(cls._tflite_model_path), exist_ok=True)
                with open(cls._tflite_model_path, 'wb') as f:
                    f.write(tflite_model)
            
            # Create TF Lite interpreter with optimizations
            cls._interpreter = tf.lite.Interpreter(
                model_content=tflite_model,
                num_threads=8  # Increased thread count for faster inference
            )
            
            # Pre-allocate tensors
            cls._interpreter.allocate_tensors()
            
            # Cache input and output details
            cls._input_details = cls._interpreter.get_input_details()
            cls._output_details = cls._interpreter.get_output_details()
            
            # Clear the original model to free memory
            cls._model = None
            
        return cls._interpreter, cls._input_details, cls._output_details
