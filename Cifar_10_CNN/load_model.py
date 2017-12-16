"""Load a model from json and run it"""
from keras.datasets import cifar10
from keras.models import model_from_json


def load_model():

    (_, _), (X_test, y_test) = cifar10.load_data()

    # load json and create model
    timestr = "12-4-17-h-m-d"  # fill the time string based on what model you want to load if it isn't the latest

    json_file = open('CNN_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("Latest_CNN.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = loaded_model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))


if __name__ == "__main__":
    load_model()
