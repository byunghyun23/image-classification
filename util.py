import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
from time import perf_counter
import os
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input


def generate_data_frame(data_dir):
    dir = Path(data_dir)
    filepaths = list(dir.glob(r'**/*.jpg'))

    labels = [str(filepaths[i]).split("\\")[-2] \
              for i in range(len(filepaths))]

    filepath = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    df = pd.concat([filepath, labels], axis=1)

    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    return df


def show_data(df):
    fig, axes = plt.subplots(nrows=4, ncols=10, figsize=(15, 7),
                            subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(df.Filepath[i]))
        ax.set_title(df.Label[i], fontsize=12)
    plt.tight_layout(pad=0.5)
    plt.show()


def show_category(df, col_name):
    vc = df[col_name].value_counts()
    plt.figure(figsize=(9, 5))
    sns.barplot(x=vc.index, y=vc, palette="rocket")
    plt.title("Number of pictures of each category", fontsize=15)
    plt.show()



def create_gen(train_df, test_df):
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.1
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='training',
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='validation',
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    return train_generator, test_generator, train_images, val_images, test_images


def get_model(model):
    kwargs = {'input_shape': (224, 224, 3),
              'include_top': False,
              'weights': 'imagenet',
              'pooling': 'avg'}

    pretrained_model = model(**kwargs)
    pretrained_model.trainable = False

    inputs = pretrained_model.input

    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    outputs = tf.keras.layers.Dense(8, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_results_of_models(models, train_images, val_images, test_images, test_label):
    for name, model in models.items():
        m = get_model(model['model'])
        models[name]['model'] = m

        start = perf_counter()

        history = m.fit(train_images, validation_data=val_images, epochs=1, verbose=0)

        duration = perf_counter() - start
        duration = round(duration, 2)
        models[name]['perf'] = duration
        print(f'** {name:20} trained in {duration} sec **')

        val_acc = history.history['val_accuracy']
        models[name]['val_acc'] = [round(v, 4) for v in val_acc]
    print()

    for name, model in models.items():
        pred = models[name]['model'].predict(test_images)
        pred = np.argmax(pred, axis=1)

        labels = (train_images.class_indices)
        labels = dict((v, k) for k, v in labels.items())
        pred = [labels[k] for k in pred]

        y_test = list(test_label)
        acc = accuracy_score(y_test, pred)
        models[name]['acc'] = round(acc, 4)
        print(f'** {name} has a {acc * 100:.2f}% accuracy on the test set **')
    print()

    # Create a DataFrame with the results
    models_result = []

    for name, v in models.items():
        models_result.append([name, models[name]['val_acc'][-1],
                              models[name]['acc'],
                              models[name]['perf']])

    results_df = pd.DataFrame(models_result,
                              columns=['model', 'val_accuracy', 'accuracy', 'Training time (sec)'])
    results_df.sort_values(by=['val_accuracy', 'accuracy', 'Training time (sec)'], ascending=[False, False, True], inplace=True)
    results_df.reset_index(inplace=True, drop=True)

    return results_df


def show_dataframe(df, x, y, title):
    plt.figure(figsize=(15, 8))
    sns.barplot(x=x, y=y, data=df)
    plt.title(title, fontsize=15)
    # plt.ylim(0,1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def load_data(images_dir, start=0, end=10000):
    IMAGE_SIZE = (224, 224)

    name_list = []
    img_list = []

    files = os.listdir(images_dir)

    cnt = 0
    for file in files[start:end]:
        try:
            path = images_dir + '/' + file

            img = image.load_img(path, target_size=IMAGE_SIZE)
            img = image.img_to_array(img)

            name_list.append(file)
            img_list.append(img)

            cnt += 1
            print(cnt, 'path: %s \t shape: %s' % (path, str(img.shape)))
            del img, file

        except FileNotFoundError as e:
            print('ERROR : ', e)

    names = np.array(name_list)
    imgs = np.stack(img_list)

    imgs = preprocess_input(imgs)

    return names, imgs