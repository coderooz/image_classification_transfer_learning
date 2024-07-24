# Image Classification with Transfer Learning

This project utilizes transfer learning to classify images into categories using a pre-trained VGG16 model. Transfer learning allows us to leverage the features learned by the VGG16 model on the ImageNet dataset to improve classification performance on a new dataset.

## Project Structure

- `data/`: Contains scripts for loading and preprocessing image data.
- `model/`: Contains the transfer learning model definition.
- `scripts/`: Contains scripts for training and evaluating the model.
- `requirements.txt`: Lists the required Python packages.

 ## Getting Started

[**Open Colab file**](https://colab.research.google.com/drive/1MQQoXCuGdUlukvAmHI5p_1ylPL3F4PEb?usp=sharing)

**OR**

 1. **Clone the repository:**
    ```bash
    git clone https://github.com/coderooz/image_classification_transfer_learning.git
    cd image_classification_transfer_learning
    ```

 2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

 3. **Prepare your dataset:**
    Place your image data in `data/train/` and `data/validation/` directories, with subdirectories for each class.

 4. **Train the model:**
    ```bash
    python scripts/train_model.py
    ```

 5. **Evaluate the model:**
    ```bash
    python scripts/evaluate_model.py
    ```

 ## Results
 The model's accuracy on the validation set will be printed after evaluation.

 ## License
 This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

 ## Acknowledgements
 - The VGG16 model is provided by [TensorFlow](https://www.tensorflow.org/).

## Contact
- Ranit Saha - [Coderooz](https://github.com/coderooz)
