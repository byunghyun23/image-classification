import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
import pandas as pd
import click
from util import generate_data_frame, show_data, show_category, show_dataframe, \
    create_gen, get_results_of_models


@click.command()
@click.option('--data_dir', default='data/train', help='Data path')
@click.option('--batch_size', default=32, help='Batch size')
@click.option('--epochs', default=300, help='Epochs')
def run(data_dir, batch_size, epochs):
    df = generate_data_frame(data_dir)
    print(df.head(5))

    print(f'Number of pictures: {df.shape[0]}')
    print(f'Number of different labels: {len(df.Label.unique())}')
    print(f'Labels: {df.Label.unique()}')
    print('Label Counts')
    print(df.Label.value_counts())

    show_data(df)

    show_category(df, 'Label')

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=0)
    print(train_df.shape, test_df.shape)

    train_generator, test_generator, train_images, val_images, test_images = create_gen(train_df, test_df)

    # Pre-trained models
    models = {
        "DenseNet121": {"model": tf.keras.applications.DenseNet121, "perf": 0},
        "MobileNetV2": {"model": tf.keras.applications.MobileNetV2, "perf": 0},
        "DenseNet201": {"model": tf.keras.applications.DenseNet201, "perf": 0},
        "EfficientNetB0": {"model": tf.keras.applications.EfficientNetB0, "perf": 0},
        "EfficientNetB1": {"model": tf.keras.applications.EfficientNetB1, "perf": 0},
        "InceptionV3": {"model": tf.keras.applications.InceptionV3, "perf": 0},
        "MobileNetV3Large": {"model": tf.keras.applications.MobileNetV3Large, "perf": 0},
        "ResNet152V2": {"model": tf.keras.applications.ResNet152V2, "perf": 0},
        "ResNet50": {"model": tf.keras.applications.ResNet50, "perf": 0},
        "ResNet50V2": {"model": tf.keras.applications.ResNet50V2, "perf": 0},
        "VGG19": {"model": tf.keras.applications.VGG19, "perf": 0},
        "VGG16": {"model": tf.keras.applications.VGG16, "perf": 0},
        "Xception": {"model": tf.keras.applications.Xception, "perf": 0}
    }

    results_df = get_results_of_models(models, train_images, val_images, test_images, test_df.Label)
    print()
    print('========== Results of models ==========')
    print(results_df)
    results_df.to_csv('models.csv', index=False)

    show_dataframe(results_df, 'model', 'accuracy', 'Accuracy on the test set')

    show_dataframe(results_df, 'model', 'Training time (sec)', 'Training time for each model in sec')

    # Get best pre-trained model
    best_model_name = results_df['model'][0]
    model = models[best_model_name]['model']

    # Train best pre-trained model (epcohs 1 -> 5)
    history = model.fit(train_images, validation_data=val_images, batch_size=batch_size, epochs=epochs)

    pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
    plt.title("Accuracy")
    plt.show()

    pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
    plt.title("Loss")
    plt.show()

    # Save model
    model.save(best_model_name + '.h5')

    # Predict the label of the test_images
    pred = model.predict(test_images)
    pred = np.argmax(pred, axis=1)

    # Map the label
    labels = (train_images.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    with open('labels.pkl', 'wb') as f:
        pickle.dump(labels, f)

    pred = [labels[k] for k in pred]

    y_test = list(test_df.Label)

    # Show Classification Report
    class_report = classification_report(y_test, pred, zero_division=1)
    print()
    print('========== Classification Report of Best Model (%s) ==========' % best_model_name)
    print(class_report)

    cf_matrix = confusion_matrix(y_test, pred, normalize='true')
    plt.figure(figsize=(10, 7))
    sns.heatmap(cf_matrix, annot=False, xticklabels=sorted(set(y_test)), yticklabels=sorted(set(y_test)), cbar=False)
    plt.title('Normalized Confusion Matrix', fontsize=23)
    plt.xticks(fontsize=15, rotation=45)
    plt.yticks(fontsize=15, rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()
