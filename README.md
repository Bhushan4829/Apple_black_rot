# Apple Black Rot Detection using Transfer Learning

This project focuses on detecting "Apple Black Rot" using transfer learning with the DenseNet201 architecture. The model has been trained to classify apple leaves into two categories: healthy and infected with black rot. This project incorporates the concepts of data augmentation, transfer learning, and fine-tuning to build a robust classification model.

## Project Structure

- **Apple_black_rot.py**: This is the main script that contains the code for training the model using transfer learning and evaluating its performance.
- **ARTIFICIAL INTELLIGENCE FOR APPLE BLACK ROT.pdf**: A detailed document explaining the project, methodology, and results.
- **README.md**: This file that explains how to use the project.
- **Test folder**: Contains test images used for evaluation.
- **Apple_black_rot folder**: Contains subfolders `Train` and `Valid` with respective images for training and validation.

## Key Features
- **Transfer Learning**: Leveraging pre-trained DenseNet201 architecture, which is pre-trained on ImageNet, for feature extraction. This saves training time and improves accuracy.
- **Data Augmentation**: Applied techniques such as zooming, shifting, and shearing to enhance the dataset and improve generalization.
- **Early Stopping and Model Checkpointing**: Used callbacks to stop training when validation accuracy plateaus and save the best model.
- **Evaluation**: Generated accuracy, loss plots, confusion matrix, and classification report to evaluate the model's performance.

## Prerequisites

To run this project, you need the following dependencies:

- Python 3.x
- TensorFlow/Keras
- Matplotlib
- Seaborn
- Scikit-learn
- Numpy
- Pandas
- Requests

Install the required dependencies using:

```bash
pip install tensorflow matplotlib seaborn scikit-learn numpy pandas requests
```
# Running the Project
1. Prepare Data:
   - Place your training data in the Apple_black_rot/Train directory with appropriate subfolders for each class.
   - Place your test images in the Test folder
2. Train the Model: Run the Apple_black_rot.py script to train the model. The script will perform the following:
   - Load the training and validation datasets using ImageDataGenerator.
   - Fine-tune the DenseNet201 model and train the classifier on apple leaf images.
   - Save the best model based on validation accuracy.
   Use the command below to run the script:
   ```
   python Apple_black_rot.py
   ```
3. Evaluate the Model: After training, the model will evaluate the test dataset and generate the following:
   - Accuracy and loss curves for both training and validation datasets.
   - A confusion matrix and classification report to display model performance on test data.
# Key Takeaways
This project was my first experience with transfer learning. Some important lessons learned include:
- Transfer learning efficiency: Pre-trained models like DenseNet201 can significantly reduce the training time while providing excellent performance on new tasks with small datasets.
- Fine-tuning: Adjusting the last few layers of the pre-trained model to better adapt to specific tasks helps improve performance.
- Data Augmentation: Simple augmentation techniques like zooming, shifting, and shearing can significantly improve model generalization, especially when working with a limited dataset.
- Callbacks: Using early stopping and model checkpointing helps prevent overfitting and ensures that the best model is saved during training.
# Results
- The model achieved high accuracy on the test set and successfully differentiated between healthy and infected apple leaves.
- The confusion matrix and classification report indicate that the model performs well in both categories.
# Future Work
- Model Improvement: Experiment with other pre-trained models like ResNet, EfficientNet, and MobileNet to see if further accuracy gains can be achieved.
- Dataset Expansion: Collect more apple leaf images from various sources to improve model robustness.
- Deployment: Consider deploying the trained model as a web service for real-time disease detection in apple orchards.
