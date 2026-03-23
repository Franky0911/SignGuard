# Traffic Sign Recognition System

This is a traffic sign recognition project that can detect signs in images, classify them, and check if they've been tampered with. I built it using YOLO for detection/classification and a ResNet-based model for detecting fake or modified signs.

## What does it do?

The system has two main parts:
1. **Sign Detection & Classification** - Uses YOLO to find traffic signs in photos and tell you what they are (speed limits, stop signs, etc.)
2. **Tamper Detection** - Checks if a sign has been vandalized, modified, or looks fake

I also added explainable AI features so you can see what parts of the image the models are actually looking at when making decisions.

## Why I built this

Traffic sign recognition is important for self-driving cars and driver assistance systems. But there's also the security angle - adversarial attacks on traffic signs are a real concern. Someone could modify a stop sign in a way that tricks an AI but looks normal to humans. This project tries to address both problems.

## Project Structure

Here's what's in the repo:

```
TSR/
├── app/                    # Core application code
├── config/                 # YAML configs for training
├── dataset/                # Training data
├── models/                 # Saved model weights
├── trainers/               # Training logic for both models
├── utils/                  # Helper functions and data loaders
├── xai/                    # Explainable AI stuff (Grad-CAM++, Integrated Gradients)
├── tools/                  # Data preprocessing and augmentation
├── streamlit_app.py        # The web interface
├── training.py             # Main training script
└── evaluate_tsr_system.py # Testing and evaluation
```

## Main Files

### streamlit_app.py
This is the web interface. You can upload images or use your camera to capture traffic signs, and it'll show you the detection results along with visual explanations of what the AI is focusing on. It's built with Streamlit because I wanted something quick and functional.

### training.py
The training pipeline. It handles training both models - first the YOLO detector, then generates a dataset of tampered signs, and finally trains the tamper detection model. You can skip steps if you already have trained models.

### evaluate_tsr_system.py
For testing the system. It'll run images through both models and generate performance metrics. You can evaluate the whole dataset or just test specific images with XAI visualizations.

## The Models

**YOLO (YOLOv8/v11)** - Does both detection and classification in one shot. It finds all traffic signs in an image and tells you what each one is. I'm using a pretrained model and fine-tuning it on traffic sign data. Works with 43 different sign classes.

**ResNet18** - Binary classifier that checks if a sign is clean or tampered. I trained this from scratch using a dataset of real signs plus artificially generated tampered versions (graffiti, overlays, blur effects, etc.).

## XAI (Explainable AI)

I implemented two techniques to visualize what the models are looking at:

- **Grad-CAM++** - Creates heatmaps showing which parts of the image are important for the prediction
- **Integrated Gradients** - More precise pixel-level attribution

Both are useful for debugging and understanding when/why the models make mistakes.

## Data Processing

### tools/preprocess.py
Takes the raw GTSRB dataset and converts it to YOLO format. Also handles train/val/test splits and generates class distribution stats.

### tools/augmentation.py
Creates augmented versions of the training images - different lighting, weather effects, blur, noise, occlusion, etc. This helps the model generalize better. I used Albumentations for most of the transforms.

### utils/tampered_data_generator.py
Generates fake tampered signs for training. It adds:
- Random graffiti (lines, circles, scribbles)
- Semi-transparent overlays
- Text vandalism
- Various blur effects

The goal is to make the tamper detector robust to different types of modifications.

## How to Use

Install dependencies:
```bash
pip install -r requirements.txt
```

Preprocess the data (if starting from scratch):
```bash
python tools/preprocess.py
python tools/augmentation.py
python setup_tampered_dataset.py
```

Train the models:
```bash
python training.py
```

Run the web app:
```bash
streamlit run streamlit_app.py
```

Test the system:
```bash
python evaluate_tsr_system.py
```

## Dataset

Using the GTSRB (German Traffic Sign Recognition Benchmark) dataset for training. After augmentation, ends up being around 20k+ images across 43 classes. The tampered signs dataset is generated synthetically.

## Dependencies

Main libraries:
- PyTorch (deep learning)
- Ultralytics (YOLO implementation)
- Streamlit (web interface)
- OpenCV (image processing)
- timm (pretrained models)
- Albumentations (data augmentation)

Check `requirements.txt` for the full list.

## Training Details

The YOLO model underwent extensive training, taking approximately 14 hours utilizing the highest available GPU compute tier on Google Colab Pro. The ResNet18 tamper detector also involves intensive training, incorporating significant regularization techniques (dropout, weight decay, label smoothing) to effectively prevent overfitting on the synthetically generated tampered datasets.

To optimize the training pipeline, I implemented mixed precision training to accelerate processing times, alongside gradient clipping to maintain stability during the runs. Early stopping is also configured to halt training efficiently once the model ceases to improve, ensuring no compute resources are wasted.

## Results

The detection model gets pretty good accuracy on the test set. The tamper detector is harder to evaluate since real-world tampered signs are rare, but it performs well on the synthetic test set and seems to generalize okay to real vandalized signs I tested.

The XAI visualizations are helpful for debugging - you can see when the model is focusing on the wrong parts of the image.

## Known Issues

- The tamper detector might be overfit to the types of tampering I generated. Real-world adversarial attacks could look different.
- YOLO sometimes misses signs that are very small or partially occluded
- The web app can be slow on CPU-only systems

## Future Plans

Things I might add:
- Support for more sign types and international signs
- Video processing (currently only does images)
- Better mobile support
- API endpoints for integration with other systems
- More sophisticated tampering detection (adversarial example detection)

## License

This is an educational/research project. Feel free to use and modify as needed.

## Credits

- YOLO by Ultralytics
- GTSRB dataset creators
- PyTorch team
- Streamlit for making web apps easy
