# Potato-Leaf-Disease-Classification
This project is aimed at developing a Deep Learning model using Convolutional Neural Networks (CNN) to classify different diseases in potato plants. The model is trained on a large dataset of potato plant images, which includes healthy potato plants as well as potato plants infected with various diseases. The trained model can accurately classify potato plants as healthy or diseased based on the input images.

Dataset
The dataset used for training the model consists of images of potato plants, which are divided into different categories based on their health status. The dataset includes the following categories:

1.Healthy Potato Plants
2.Late Blight Disease
3.Early Blight Disease
4.Leaf Mold Disease
5.Common Scab Disease
The dataset is divided into training, validation, and testing sets. The training set is used to train the model, the validation set is used to tune the hyperparameters and avoid overfitting, and the testing set is used to evaluate the final model's performance.

Model Architecture
The deep learning model uses Convolutional Neural Networks (CNN) to classify the potato plant images. The CNN architecture consists of multiple convolutional layers, followed by pooling layers, and fully connected layers. The activation function used in the CNN is ReLU (Rectified Linear Unit), and the output layer uses a softmax activation function to predict the class probabilities.

The model is implemented using Python programming language and popular deep learning libraries such as TensorFlow and Keras.
Dataset: https://www.kaggle.com/code/tharunk07/potatoleaves-disease-prediction/input

Conclusion
The Potato Disease Classification project is a practical application of Deep Learning and Convolutional Neural Networks (CNN) for detecting diseases in potato plants. The trained model can accurately classify potato plants as healthy or diseased, which can help farmers take timely action to prevent crop losses. The project can be extended by incorporating real-time image processing, deploying the model on edge devices, or
