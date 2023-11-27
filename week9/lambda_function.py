import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor


# url = "https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg"

classes = ["bee", "wasp"]

preprocess = create_preprocessor("xception", target_size=(150, 150))

interpreter = tflite.Interpreter(model_path="bees-wasps.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]


def predict(url):

    X = preprocess.from_url(url)

    interpreter.set_tensor(input_index, X) # Set the value of the input tensor
    interpreter.invoke() # Run the model
    preds = interpreter.get_tensor(output_index) # Get the output tensor

    float_prediction = preds[0].tolist()

    return dict(zip(classes, float_prediction))


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result



