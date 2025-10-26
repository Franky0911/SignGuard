"""
TRAINING PIPELINE - THE BRAIN TRAINING CENTER
==============================================
This is the main training script that teaches all our AI models how to recognize traffic signs.

What does training mean?
- Like teaching a child to recognize animals by showing many pictures
- We show the AI thousands of traffic sign images
- Over time, it learns the patterns that make each sign unique

This script trains THREE different AI models:
1. DETECTION MODEL (YOLO) - Learns to find/locate traffic signs in images
   - Like teaching: "Find all the signs in this photo"
   
2. CLASSIFICATION MODEL - Learns to identify what TYPE of sign it is
   - Like teaching: "This is a stop sign, this is a speed limit, etc."
   
3. TAMPERING DETECTOR - Learns to spot fake or modified signs
   - Like teaching: "This sign looks suspicious, it might be fake"

The training happens in order because each model needs the previous one.
Think of it as learning to walk before you run!
"""

# Import Python libraries for file and time operations
import os  # For file system operations
import time  # For tracking how long training takes
from pathlib import Path  # For handling file paths

# Import our custom training modules - each one trains a different AI model
from trainers import DetectionTrainer, ClassifierTrainer  # Train detection and classification
from trainers.tampered_detector_trainer import TamperedDetectorTrainer  # Train tampering detector
from utils.tampered_data_generator import TamperedSignGenerator  # Creates fake/tampered signs for training
from utils.config_loader import ConfigLoader  # Loads training settings from configuration files


class TrainingPipeline:
    """
    TRAINING PIPELINE ORCHESTRATOR
    ===============================
    This class is like a project manager that coordinates the training of all three models.
    It makes sure everything is done in the right order and with the right settings.
    
    What it does:
    1. Loads configuration settings (how many training iterations, learning rate, etc.)
    2. Sets up the computing device (GPU if available, otherwise CPU)
    3. Creates trainers for each of the three models
    4. Runs the complete training pipeline
    
    Think of this as the conductor of an orchestra - making sure all parts work together!
    """
    
    def __init__(self, config_path='config/training_config.yaml'):
        """
        Initialize the training pipeline
        
        Args:
            config_path: Path to YAML file containing training settings
                        (like a recipe card with all the ingredients and steps)
        """
        # Load configuration settings from YAML file
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        
        # Setup computing device (GPU is much faster than CPU for AI training)
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")  # Tell user if using GPU or CPU
        # Optional cuDNN benchmark for faster convolutions on fixed input sizes
        if self.device.type == 'cuda':
            try:
                import torch.backends.cudnn as cudnn
                cudnn.benchmark = bool(self.config.get('detection', {}).get('benchmark', True))
                cudnn.deterministic = bool(self.config.get('detection', {}).get('deterministic', False))
                print(f"cuDNN benchmark: {cudnn.benchmark}, deterministic: {cudnn.deterministic}")
            except Exception as _e:
                print("cuDNN optimization not applied:", _e)
        
        # Validate and create required paths
        self.config_loader.validate_paths()
        
        # Initialize trainers
        self.detection_trainer = DetectionTrainer(self.config)
        self.tampered_detector_trainer = TamperedDetectorTrainer(self.config, self.device)
        self.classifier_trainer = ClassifierTrainer(self.config, self.device)
        self.tampered_generator = TamperedSignGenerator()
    
    
    def run_full_training(self, data_yaml):
        """Run complete training pipeline"""
        print("🚀 Starting Full Training Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        pipeline_summary = {
            'detection': {'status': 'pending', 'skipped': False},
            'tampered_data': {'status': 'pending', 'skipped': False},
            'tampered_detector': {'status': 'pending', 'skipped': False},
            'classifier': {'status': 'pending', 'skipped': False}
        }
        
        # Check which models already exist
        print("\n🔍 Checking for existing models...")
        detection_model_path = 'models/traffic_sign_detection.pt'
        tampered_detector_model_path = 'models/tampered_sign_detector.pth'
        classifier_model_path = 'models/traffic_sign_classifier.pth'
        
        if os.path.exists(detection_model_path):
            print(f"✅ Detection model found: {detection_model_path}")
            pipeline_summary['detection']['skipped'] = True
        else:
            print(f"❌ Detection model not found: {detection_model_path}")
            
        if os.path.exists(tampered_detector_model_path):
            print(f"✅ Tampered detector model found: {tampered_detector_model_path}")
            pipeline_summary['tampered_detector']['skipped'] = True
        else:
            print(f"❌ Tampered detector model not found: {tampered_detector_model_path}")
            
        if os.path.exists(classifier_model_path):
            print(f"✅ Classifier model found: {classifier_model_path}")
            pipeline_summary['classifier']['skipped'] = True
        else:
            print(f"❌ Classifier model not found: {classifier_model_path}")
        
        # 1. Train detection model
        print("\n1️⃣ Training Detection Model...")
        if pipeline_summary['detection']['skipped']:
            print("⏭️  Skipping detection training - model already exists")
            detection_results = {'status': 'skipped', 'model_path': detection_model_path}
        else:
            detection_results = self.detection_trainer.train(data_yaml)
        pipeline_summary['detection']['status'] = detection_results.get('status', 'unknown')
        
        # 2. Generate tampered signs dataset
        print("\n2️⃣ Generating Tampered Signs Dataset...")
        tampered_dataset_path = Path('dataset/tampered_signs')
        
        # Check if tampered dataset already exists
        if (tampered_dataset_path / 'train' / 'real').exists() and \
           (tampered_dataset_path / 'train' / 'tampered').exists():
            print("✅ Tampered dataset already exists, skipping generation")
            tampered_data_results = {'status': 'skipped', 'path': str(tampered_dataset_path)}
        else:
            print("🔧 Generating tampered signs from real traffic signs...")
            tampered_data_results = self.tampered_generator.generate_dataset(
                input_dir='dataset/augmented/train/images',
                output_dir=str(tampered_dataset_path),
                num_tampered_per_image=3,
                test_split=0.2
            )
        pipeline_summary['tampered_data']['status'] = tampered_data_results.get('status', 'completed') if isinstance(tampered_data_results, dict) else 'completed'
        
        # 3. Train tampered detector
        print("\n3️⃣ Training Tampered Sign Detector...")
        if pipeline_summary['tampered_detector']['skipped']:
            print("⏭️  Skipping tampered detector training - model already exists")
            tampered_detector_results = {'status': 'skipped', 'model_path': tampered_detector_model_path}
        else:
            # Use the generated tampered dataset
            real_path = str(tampered_dataset_path)  # Will auto-detect train/real structure
            tampered_path = str(tampered_dataset_path)  # Will auto-detect train/tampered structure
            
            tampered_detector_results = self.tampered_detector_trainer.train(
                real_path,
                tampered_path
            )
        pipeline_summary['tampered_detector']['status'] = tampered_detector_results.get('status', 'unknown')
        
        # 4. Train classifier
        print("\n4️⃣ Training Traffic Sign Classifier...")
        if pipeline_summary['classifier']['skipped']:
            print("⏭️  Skipping classifier training - model already exists")
            classifier_results = {'status': 'skipped', 'model_path': classifier_model_path}
        else:
            classifier_results = self.classifier_trainer.train(
                'dataset/augmented'  # Use augmented dataset for better generalization
            )
        pipeline_summary['classifier']['status'] = classifier_results.get('status', 'unknown')
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Print pipeline summary
        print(f"\n🎉 Training Pipeline Completed!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"📁 Models saved in: models/")
        
        print(f"\n📊 Pipeline Summary:")
        print("=" * 40)
        for step, info in pipeline_summary.items():
            status_emoji = "✅" if info['status'] == 'completed' else "⏭️" if info['skipped'] else "❌"
            print(f"{status_emoji} {step.replace('_', ' ').title()}: {info['status']}")
        
        return {
            'detection_results': detection_results,
            'tampered_data_results': tampered_data_results,
            'tampered_detector_results': tampered_detector_results,
            'classifier_results': classifier_results,
            'total_time': total_time,
            'pipeline_summary': pipeline_summary
        }


def main():
    """Main training function"""
    # Initialize training pipeline
    pipeline = TrainingPipeline()
    
    # Run full training
    results = pipeline.run_full_training('dataset/augmented/augmented_config.yaml')
    
    print("\n" + "="*60)
    print("🎉 Training completed successfully!")
    print("Models are ready for inference and dashboard integration.")
    print("="*60)


if __name__ == "__main__":
    main()
