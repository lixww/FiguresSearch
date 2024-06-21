#%%
# Prerequists
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf

from deepface.basemodels import Facenet

#%%
model  = Facenet.load_facenet512d_model()

converter = tf.lite.TFLiteConverter.from_keras_model(model)

print(tf.__version__)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
print('--- ---')
tflite_model = converter.convert()
print('--- ---')

with open('test_models/facenet512_weights.tflite', 'wb') as f:
  f.write(tflite_model)

#%%
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="test_models/facenet512_weights.tflite")
interpreter.allocate_tensors()
#%%
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#%%
# Print input details
print("Input Details:")
for input_detail in input_details:
    print(input_detail)

# Print output details
print("\nOutput Details:")
for output_detail in output_details:
    print(output_detail)

# You can also print the tensor details to see the shape and type of all tensors in the model
print("\nTensor Details:")
for tensor_detail in interpreter.get_tensor_details():
    print(tensor_detail)



