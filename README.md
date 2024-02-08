![app · Streamlit - Google Chrome 08_02_2024 7_24_46 pm](https://github.com/chussboi96/Medicinal-Plant-Leaf-detection-using-computer-vision/assets/103335581/f0ce8bbd-3d93-413c-9c53-583b0141cf77)![app · Streamlit - Google Chrome 08_02_2024 7_23_42 pm](https://github.com/chussboi96/Medicinal-Plant-Leaf-detection-using-computer-vision/assets/103335581/df3f6a14-4b00-40bc-99c2-7bf7aeefb971)# Medicinal-Plant-Leaf-detection-using-computer-vision
An AI-powered application using TensorFlow and ResNet50V2 for identifying and classifying medicinal plant leaves.

https://drive.google.com/file/d/1gBPA8GR99ltHsrMvvhMTJ87y3aBh3nGl/view?usp=drive_link
This is the google-drive link for the model used in the streamlit application with an accuracy of 94%.

This repository contains the implementation of a deep learning model for medicinal plant leaf detection. The model is built using TensorFlow and Keras, leveraging the pre-trained ResNet50V2 architecture. The primary objective is to accurately classify different types of medicinal plant leaves. The model is trained using a categorical cross-entropy loss function and an Adam optimizer. Training and validation processes are visualized for accuracy and loss.

Features:
- Utilizes the powerful ResNet50V2 model pre-trained on ImageNet data.
- Custom layers added on top of ResNet50V2 for specific classification tasks.
- Implements data augmentation techniques to improve model robustness.
- Includes a detailed analysis of training and validation accuracy and loss.
- The dataset comprises segmented images of various medicinal plant leaves.
- It's hosted on Google Drive and accessed directly in the notebook. (https://drive.google.com/drive/folders/1IcHbG_k3DpMcd6JeFwnJf4y8-9QWNdYM?usp=drive_link)

How to Use:
- Clone the repository.
- Ensure you have the required libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, and Streamlit.
- Run the Jupyter Notebook to train the model or download from the provided link.
- Use the Streamlit app for interactive leaf classification.

Streamlit App
An interactive web application created with Streamlit allows users to upload a leaf image and receive a prediction. The app utilizes the trained model for classification.


Note:
The dataset path and Streamlit app configurations might need adjustments based on your setup.

![app · Streamlit - Google Chrome 08_02_2024 7_23_42 pm](https://github.com/chussboi96/Medicinal-Plant-Leaf-detection-using-computer-vision/assets/103335581/5b988432-84d4-4c75-a40c-7a2bfa76921e)
![app · Streamlit - Google Chrome 08_02_2024 7_24_46 pm](https://github.com/chussboi96/Medicinal-Plant-Leaf-detection-using-computer-vision/assets/103335581/d11a79f2-0272-4e09-8bde-9e0be6b6dd76)


