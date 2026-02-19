This project is a sports person classifier that uses computer vision and machine learning to identify different sports figures from images. The model is trained on a dataset of images and uses a Support Vector Classifier (SVC) to make predictions.

Technologies Used
opencv-python (OpenCV):** Used for computer vision tasks such as image processing, face detection, and eye detection using pre-trained Haar Cascades. numpy:** Essential for numerical operations, especially for handling image data as arrays. matplotlib:** Used for visualizing images and plotting the confusion matrix to evaluate model performance. scikit-learn:** Provides the machine learning tools, including the SVC for classification, GridSearchCV for hyperparameter tuning, and performance metrics like the classification report and confusion matrix. joblib:** Used to save and load the trained machine learning model (saved_model.pkl). json:** Used to save the class dictionary that maps class names to numerical labels (class_dictionary.json).

Methodology
The project follows a standard machine learning pipeline for image classification:

Data Preprocessing and Augmentation:

The notebook reads images from a local dataset. An example image from the "Ma_Long" directory is loaded.
It uses OpenCV's Haar Cascades to detect and crop faces and eyes from the images.
Processed images are resized to a uniform size (32x32 pixels).
A combination of the cropped face and both eyes is used to form a single feature vector using np.hstack.
Model Training:

The model uses a Support Vector Classifier (SVC) with a rbf (Radial Basis Function) kernel.
GridSearchCV is used to find the best hyperparameters for the SVC, which helps in optimizing the model's accuracy.
Evaluation:

The model's performance is evaluated using a classification report, which provides precision, recall, and F1-score for each class.
A confusion matrix is generated to visualize the model's predictions against the actual labels.
Model Persistence:

After training, the best-performing model (best_clf) is saved to a file named saved_model.pkl using joblib.
The dictionary mapping class names (e.g., 'lionel_messi') to numerical labels is saved to a file named class_dictionary.json using json.
How to Run the Project
Dependencies: Ensure you have the required libraries installed. The notebook shows using pip install opencv-python. You will also need numpy, matplotlib, and scikit-learn.
Dataset: Place your image dataset in a directory with subfolders for each class (e.g., a folder named dataset with subfolders for each sports person).
Run the Notebook: Execute the Sports_CLassfication_Project.ipynb notebook in a Jupyter environment. The code will handle data loading, training, and evaluation.
Use the Saved Model: You can use the saved_model.pkl and class_dictionary.json files to make predictions on new images without retraining the model.
