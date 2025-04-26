# X-Fracture: AI Fracture Detection - Deep learning based bone fracture detection in X-Rays



Below is a template for another sample project. Please follow this template.
# [DL] X-Fracture: AI Fracture Detection

## Introduction
Bone fractures are a common medical condition that needs to be diagnosed quickly and accurately in order to be treated effectively. Using radiological images, this project investigates the use of deep learning, part of AI, for the binary classification problem of bone fracture detection. Building on pre-existing convolutional neural networks (CNNs), the model uses architectural improvements to increase classification accuracy while cutting down on training time and cost. The outcomes show significant gains over the use of basic machine learning (ML) or Deep Learning(DL) techniques alone.
## Project Metadata
### Authors
- **Team:** Ahmed Baabdullah
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [Novel transfer learning based bone fracture detection using radiographic images](https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-024-01546-4)

### Reference Dataset
- [Fracture detection using x-ray images](https://www.kaggle.com/datasets/devbatrax/fracture-detection-using-x-ray-images/)


## Project Technicalities

### Terminologies
- **Mobilg:** A generative model that progressively transforms random noise into coherent data.

### Problem Statements
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.
- **Problem 3:** There is limited capability in performing style transfer and generating diverse artistic variations.

### Loopholes or Research Areas
- **testing results:** needs to propose a good archatuhure 
- **Computational Resources:** Training requires GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the code. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`Experiments_Novel_Transfer_Learning_Based_Bone_Fracture_Detection_Using_Radiographic_Images.ipnyb`**: Contains the code from paper.
- **`project testing.ipnyb`**: see what happens after changes in above
- **`Fracture_Detection_CNN_ResNet,DenseNet,MobileNet.ipnyb`**: another code for same data which doesn't freeze layer.
- **`enhanced Fracture_Detection_CNN_ResNet,DenseNet,MobileNet.ipnyb.ipnyb`**: freeze layer

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Data:** The model reads the train folder.

2. **training & setp:**
   - **models:** define the models then train them

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **results :**show different training, val and testing accruacy 

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
 **Resource Providers:** Gratitude to google colab for providing the computational resources necessary for this project.


