# Lip Reading Using Deep Learning

## Overview

Lip Verbaliser is a deep learning-based project aimed at converting lip movements into textual data. It leverages advanced computer vision techniques and sequence modeling to accurately predict spoken words by analyzing lip movements in video sequences.

## Installation and Dependencies

To run the Lip Verbaliser project, several libraries are required. Here is a list of the dependencies along with a brief explanation of their use:

- **OpenCV** ([`opencv-python`](https://pypi.org/project/opencv-python/)): 
  - Used for video processing tasks such as reading frames from video files and manipulating images.
  
- **Matplotlib** ([`matplotlib`](https://pypi.org/project/matplotlib/)): 
  - Utilized for visualizing data, particularly for plotting and displaying images.
  
- **Imageio** ([`imageio`](https://pypi.org/project/imageio/)): 
  - Facilitates reading and writing a wide range of image data, including animated sequences.
  
- **gdown** ([`gdown`](https://pypi.org/project/gdown/)): 
  - A simple Python tool to download files from Google Drive.
  
- **TensorFlow** ([`tensorflow`](https://pypi.org/project/tensorflow/)): 
  - The core library for developing and training the deep learning model.
  
- **NumPy** ([`numpy`](https://pypi.org/project/numpy/)): 
  - Essential for numerical operations and handling arrays, which are extensively used in machine learning.

To install the dependencies, run the following command:

```sh
pip install opencv-python matplotlib imageio gdown tensorflow
```
## Dataset
- You can go here and download the video dataset that I have used (It has just one speaker, 'coz eventually I grabbed data of meself and trained it on that as well..so) : https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL
- Extracted form the original GRID corpus dataset which has around 34 speakers : http://spandh.dcs.shef.ac.uk/gridcorpus/

## Data Loading and Preprocessing

The data loading functions are designed to handle video and alignment files. The preprocessing steps involve converting RGB frames to grayscale, normalizing the frames, and extracting relevant regions around the lips.

### Video Loading

```python
def load_video(path: str) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std
```

This function captures video frames, converts them to grayscale, and normalizes them to have a mean of zero and a standard deviation of one.

### Alignment Loading

```python
def load_alignments(path: str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]
```

This function reads alignment files, extracts relevant tokens (excluding silence), and converts them to numerical values using a predefined vocabulary.

## Data Pipeline

The data pipeline is constructed using TensorFlow's `tf.data.Dataset` API. It involves shuffling, mapping, batching, and prefetching operations to efficiently load and preprocess the data for training.

```python
data = tf.data.Dataset.list_files('./data/s1/*.mpg')
data = data.shuffle(500, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(2, padded_shapes=([75, None, None, None], [40]))
data = data.prefetch(tf.data.AUTOTUNE)
train = data.take(450)
test = data.skip(450)
```

## Deep Neural Network Design

The core of the Lip Verbaliser is a deep neural network composed of 3D convolutional layers and bidirectional LSTM layers. The network architecture is designed to capture spatial and temporal features from video sequences.

```python
model = Sequential()
model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))
model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))
model.add(Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))
```
### Why Bidirectional LSTM?
Bidirectional LSTMs are used because they can capture context from both past and future states, which is crucial in lip reading where understanding the sequence of movements over time is important. This ability to look at the data from both directions helps improve the accuracy of the predictions.

## Training the Model

The model is trained using the Connectionist Temporal Classification (CTC) loss function, which is suitable for sequence-to-sequence problems where alignment between input and output sequences is unknown.

```python
def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss
```

The model is compiled and trained with appropriate callbacks to save checkpoints, adjust the learning rate, and produce example outputs after each epoch.

```python
model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)

checkpoint_callback = ModelCheckpoint(os.path.join('models', 'checkpoint'), monitor='loss', save_weights_only=True)
schedule_callback = LearningRateScheduler(scheduler)
example_callback = ProduceExample(test)

model.fit(train, validation_data=test, epochs=100, callbacks=[checkpoint_callback, schedule_callback, example_callback])
```

## Making Predictions

After training, the model can be used to predict the text corresponding to lip movements in video sequences.

```python
model.load_weights('models/checkpoint')
test_data = test.as_numpy_iterator()
sample = test_data.next()

yhat = model.predict(sample[0])
decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75, 75], greedy=True)[0][0].numpy()
```
## Input and Output
### Input
- **Video Files** : The primary input is a video file containing sequences of lip movements. Each video file is typically in .mpg format. The video frames are preprocessed to grayscale and normalized.
- **Alignment Files** : These files provide the ground truth alignments of spoken words corresponding to the lip movements in the video files.

### Output
- **Predicted Text** : The output is a textual representation of the spoken words, as inferred from the lip movements in the input video sequences.
- The output is decoded from the predicted character probabilities using the CTC decoding algorithm.

## Conclusion

This project demonstrates the application of deep learning in lip reading. By combining computer vision techniques with sequence modeling, it achieves accurate text predictions from video sequences of lip movements. The project uses powerful libraries such as TensorFlow and OpenCV, and follows a structured approach to data processing, model training, and prediction.
