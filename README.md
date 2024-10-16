# Feed-Forward Neural Network on UTKFace Dataset

## Task
This project implements a simple feed-forward neural network using TensorFlow. The task is to predict the race of individuals based on facial images from the UTKFace dataset. The goal is to classify the race attribute of individuals in the dataset through neural network-based classification.

## Dataset
The UTKFace dataset contains facial images labeled with various attributes, including age, gender, and race. The images are 48x48 pixels, and the race attribute is divided into several categories:
- **Class labels**: The dataset includes different race categories, such as White, Black, Asian, Indian, and Others.
- **Size of the dataset**: The dataset consists of thousands of facial images of varying age and gender.

### Data Preprocessing
1. The dataset is split into training and test sets using a 70:30 ratio.
2. Images are resized and normalized before being fed into the neural network.

## Methods
A simple feed-forward neural network was implemented with multiple layers to perform classification:
- **Input layer**: The input to the network is a flattened vector representing the pixels of the facial images.
- **Hidden layers**: The model includes fully connected layers with activation functions such as ReLU.
- **Output layer**: The output layer uses the softmax function for classification into different race categories.

### Optimizer and Loss Function
- **Optimizer**: Adam optimizer was used with a momentum parameter to speed up convergence and avoid oscillations during training.
- **Loss Function**: Cross-entropy loss function was used as it is suitable for classification tasks.

### Regularization
- **L2 Regularization**: Applied to prevent overfitting by penalizing large weights.
- **Dropout**: A dropout layer was added to randomly drop neurons during training to avoid overfitting.

## Hyperparameter Tuning
- **Batch Size**: The batch size was tuned to find a balance between training time and model performance.
- **Learning Rate**: The learning rate was adjusted to ensure the model converges without overshooting the minimum loss.
- **Number of Epochs**: Multiple experiments were conducted with different numbers of epochs to find the optimal value where the model performs well without overfitting.

### Final Model Configuration
- **Number of layers**: The network consists of an input layer, two hidden layers with ReLU activations, and an output layer with softmax activation.
- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Epochs**: 50

## Conclusion
This project demonstrates the implementation of a simple feed-forward neural network using TensorFlow to classify race in the UTKFace dataset. Through proper data preprocessing, hyperparameter tuning, and the use of regularization techniques, the model was able to achieve reasonable classification performance.

## Course
- **Course Title**: Artificial Intelligence
