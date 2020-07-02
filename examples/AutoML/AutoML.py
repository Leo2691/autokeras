from keras.datasets import cifar10
import autokeras as ak
from tensorflow.keras.models import model_from_json
from sklearn.metrics import classification_report
import os

def build_model():
    input_layer = ak.Input()
    cnn_layer = ak.ConvBlock()(input_layer)
    cnn_layer2 = ak.ConvBlock()(cnn_layer)
    dense_layer = ak.DenseBlock()(cnn_layer2)
    dense_layer2 = ak.DenseBlock()(dense_layer)
    output_layer = ak.ClassificationHead(num_classes=10)(dense_layer2)

    automodel = ak.auto_model.AutoModel(input_layer, output_layer, max_trials=20, seed=123, project_name='autoML')

    return automodel



def main():
    print ("Loading CIFAR10 data ...")

    ((trainX, trainY), (testX, testY)) = cifar10.load_data()

    lable_names= ["airplane", "automobile", "bird", "cat", "deer",
                  "dog", "frog", "horse", "ship", "truck"]

    OUTPUT_PATH = "output"

    automodel = build_model()

    automodel.fit(trainX, trainY, validation_split=0.2, epochs=40, batch_size=64)

    score = automodel.evaluate(testX, testY)
    model = automodel.export_model()
    model_json = model.to_json()


    with open('autoML.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_waights("model.h5")

    print("Model saved to the disk")

    predicted = automodel.predict(testX)

    report = classification_report(testY, predicted, target_names=lable_names)

    p = os.path.join(os.path.dirname(OUTPUT_PATH), 'results.txt')
    f = open(p, 'w')
    f.write(report)
    f.close()

main()
