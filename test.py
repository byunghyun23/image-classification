import click
import numpy as np
import pandas as pd
import pickle
from tensorflow.python.keras.saving.save import load_model

from util import load_data


@click.command()
@click.option('--data_dir', default='data/test', help='Data path')
@click.option('--model_name', default='InceptionV3', help='Model name')
def run(data_dir, model_name):
    img_names, X_test = load_data(data_dir)
    loaded_model = load_model(model_name + '.h5')
    pred = loaded_model.predict(X_test)
    pred = np.argmax(pred, axis=1)

    with open('labels.pkl', 'rb') as f:
        labels = pickle.load(f)
        pred = [labels[k] for k in pred]

    results_df = pd.DataFrame({'image_name': img_names, 'class': pred})
    results_df.to_csv('results.csv', index=False)


if __name__ == '__main__':
    run()