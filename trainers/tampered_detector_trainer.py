"""
Tampered Sign Detector Trainer
Handles training of the tampered sign detection model with regularization
to prevent overfitting and ensure good generalization
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from models import ModelFactory
from utils.data_utils import get_tampered_detection_loaders


class TamperedDetectorTrainer:
    """Handles tampered sign detection model training with regularization"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.tampered_detector = None
    
    def train(self, real_path, tampered_path):
        """
        Train the tampered detector model
        
        Args:
            real_path: Path to directory containing real/clean traffic signs
            tampered_path: Path to directory containing tampered traffic signs
        
        Returns:
            Dictionary with training results and status
        """
        model_path = 'models/tampered_sign_detector.pth'
        
        # Check if model already exists
        if os.path.exists(model_path):
            print("✅ Tampered detector already exists, skipping training")
            print(f"📁 Model found at: {model_path}")
            return {'status': 'skipped', 'model_path': model_path}
        
        print("🚀 Training Tampered Sign Detector")
        print("=" * 50)
        print("🔧 Regularization techniques enabled:")
        print("   - Dropout layers")
        print("   - Weight decay (L2 regularization)")
        print("   - Data augmentation")
        print("   - Early stopping")
        print("   - Learning rate scheduling")
        print("   - Gradient clipping")
        print("=" * 50)
        
        try:
            # Create tampered detector
            # Use tampered_detection config if available, fallback to fake_detection
            detector_config = self.config.get('tampered_detection', 
                                             self.config.get('fake_detection', {}))
            
            self.tampered_detector = ModelFactory.create_fake_detector(
                detector_config.get('model', 'resnet18')
            )
            self.tampered_detector.to(self.device)
            
            # Count parameters
            total_params = sum(p.numel() for p in self.tampered_detector.parameters())
            trainable_params = sum(p.numel() for p in self.tampered_detector.parameters() if p.requires_grad)
            print(f"📊 Model parameters: {trainable_params:,} trainable / {total_params:,} total")
            
            # Create data loaders with augmentation
            dl_cfg = self.config.get('data', {})
            train_loader, val_loader = get_tampered_detection_loaders(
                real_path, tampered_path, 
                batch_size=detector_config.get('batch_size', 32),
                num_workers=dl_cfg.get('num_workers', 2),
                pin_memory=dl_cfg.get('pin_memory', True),
                persistent_workers=dl_cfg.get('persistent_workers', True),
                prefetch_factor=dl_cfg.get('prefetch_factor', 2)
            )
            
            if train_loader is None:
                print("⚠️  No data available for tampered detection training")
                return {'status': 'failed', 'error': 'No data available'}
            
            print(f"📊 Training batches: {len(train_loader)}")
            print(f"📊 Validation batches: {len(val_loader)}")
            
            # Calculate class weights to handle imbalance
            # Access the underlying dataset (handle both Subset and direct Dataset)
            train_dataset = train_loader.dataset
            if hasattr(train_dataset, 'dataset'):
                # It's a Subset, get the underlying dataset
                samples = train_dataset.dataset.samples
                if hasattr(train_dataset, 'indices'):
                    # Filter by indices
                    samples = [samples[i] for i in train_dataset.indices]
            else:
                samples = train_dataset.samples
            
            # Count classes
            real_count = len([s for s in samples if s[1] == 0])
            tampered_count = len([s for s in samples if s[1] == 1])
            total_count = real_count + tampered_count
            
            # Calculate inverse frequency weights
            class_weights = torch.tensor([
                total_count / (2 * real_count) if real_count > 0 else 1.0,
                total_count / (2 * tampered_count) if tampered_count > 0 else 1.0
            ], dtype=torch.float32).to(self.device)
            
            print(f"📊 Class distribution: Real={real_count}, Tampered={tampered_count}")
            print(f"🔧 Class weights: Real={class_weights[0]:.4f}, Tampered={class_weights[1]:.4f}")
            print(f"🔧 Imbalance ratio: 1:{tampered_count/real_count:.2f} (Real:Tampered)")
            
            # Training setup with regularization
            # Label smoothing for better generalization
            label_smoothing = detector_config.get('label_smoothing', 0.1)
            criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                weight=class_weights  # Add class weights to handle imbalance
            )
            print(f"🔧 Label smoothing: {label_smoothing}")
            
            # Optimizer with weight decay (L2 regularization)
            lr = float(detector_config.get('learning_rate', 0.001))
            weight_decay = float(detector_config.get('weight_decay', 0.0001))
            print(f"🔧 Learning rate: {lr}")
            print(f"🔧 Weight decay (L2): {weight_decay}")
            
            optimizer = optim.AdamW(
                self.tampered_detector.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            # Learning rate scheduler - CosineAnnealingLR with warm restarts
            scheduler_type = detector_config.get('scheduler', 'cosine')
            if scheduler_type == 'cosine':
                sp = detector_config.get('scheduler_params', {})
                T_max = int(sp.get('T_max', 10))
                eta_min = float(sp.get('eta_min', 1e-6))
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=T_max,
                    eta_min=eta_min
                )
                print(f"🔧 Scheduler: CosineAnnealingLR (T_max={T_max}, eta_min={eta_min})")
            else:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode='min', 
                    factor=0.1, 
                    patience=3, 
                    verbose=True
                )
                print(f"🔧 Scheduler: ReduceLROnPlateau")
            
            # Mixed precision training for efficiency
            use_amp = torch.cuda.is_available() and detector_config.get('use_amp', True)
            scaler = torch.amp.GradScaler('cuda') if use_amp else None
            if use_amp:
                print("🔧 Automatic Mixed Precision (AMP): Enabled")
            
            # Gradient clipping to prevent exploding gradients
            grad_clip_norm = float(detector_config.get('grad_clip_norm', 1.0))
            print(f"🔧 Gradient clipping: {grad_clip_norm}")
            
            # Training loop
            epochs = int(detector_config.get('epochs', 20))
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []
            
            best_val_loss = float('inf')
            best_val_acc = 0.0
            patience = int(detector_config.get('early_stopping_patience', 7))
            epochs_without_improve = 0
            best_model_path = Path('models') / 'tampered_sign_detector.best.pth'
            
            print(f"\n🎯 Training for {epochs} epochs (early stopping patience: {patience})")
            print("=" * 50)

            for epoch in range(epochs):
                # Training phase
                self.tampered_detector.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    if scaler:
                        with torch.amp.autocast('cuda'):
                            output = self.tampered_detector(data)
                            loss = criterion(output, target)
                        
                        scaler.scale(loss).backward()
                        
                        # Gradient clipping
                        if grad_clip_norm > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.tampered_detector.parameters(), 
                                grad_clip_norm
                            )
                        
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        output = self.tampered_detector(data)
                        loss = criterion(output, target)
                        loss.backward()
                        
                        # Gradient clipping
                        if grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.tampered_detector.parameters(), 
                                grad_clip_norm
                            )
                        
                        optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    train_total += target.size(0)
                    train_correct += (predicted == target).sum().item()
                    
                    # Print progress every 10 batches
                    if (batch_idx + 1) % 10 == 0:
                        print(f'  Batch [{batch_idx+1}/{len(train_loader)}] '
                              f'Loss: {loss.item():.4f}', end='\r')
                
                # Validation phase
                print()  # New line after training progress
                print('  🔍 Running validation...', end='', flush=True)
                self.tampered_detector.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(val_loader):
                        data, target = data.to(self.device), target.to(self.device)
                        
                        if scaler:
                            with torch.amp.autocast('cuda'):
                                output = self.tampered_detector(data)
                                loss = criterion(output, target)
                        else:
                            output = self.tampered_detector(data)
                            loss = criterion(output, target)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(output.data, 1)
                        val_total += target.size(0)
                        val_correct += (predicted == target).sum().item()
                        
                        # Show validation progress
                        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
                            print(f'\r  🔍 Validating... [{batch_idx+1}/{len(val_loader)}]', end='', flush=True)
                
                # Calculate metrics
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                train_acc = 100. * train_correct / train_total
                val_acc = 100. * val_correct / val_total
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)
                
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f'\n📊 Epoch [{epoch+1}/{epochs}]:')
                print(f'   Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'   Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print(f'   Learning Rate: {current_lr:.6f}')
                
                # Check for overfitting
                if avg_train_loss < avg_val_loss:
                    gap = avg_val_loss - avg_train_loss
                    if gap > 0.3:
                        print(f'   ⚠️  Potential overfitting detected (gap: {gap:.4f})')
                
                # Update scheduler
                if scheduler_type == 'cosine':
                    scheduler.step()
                else:
                    scheduler.step(avg_val_loss)
                
                # Early stopping and best checkpoint (based on validation accuracy)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_loss = avg_val_loss
                    epochs_without_improve = 0
                    Path('models').mkdir(exist_ok=True)
                    torch.save(self.tampered_detector.state_dict(), best_model_path)
                    print(f'   ✅ New best model! Val Acc: {best_val_acc:.2f}%, Val Loss: {best_val_loss:.4f}')
                    print(f'   💾 Checkpoint saved to {best_model_path}')
                else:
                    epochs_without_improve += 1
                    print(f'   📉 No improvement for {epochs_without_improve} epoch(s)')
                    
                    if epochs_without_improve >= patience:
                        print(f'\n⏹️  Early stopping triggered at epoch {epoch+1}')
                        print(f'   Best Val Acc: {best_val_acc:.2f}%')
                        print(f'   Best Val Loss: {best_val_loss:.4f}')
                        break
                
                print('-' * 50)
            
            # Load best model
            self.tampered_detector.load_state_dict(torch.load(best_model_path))
            
            # Save final model
            Path('models').mkdir(exist_ok=True)
            torch.save(self.tampered_detector.state_dict(), model_path)
            
            print("\n" + "=" * 50)
            print("✅ Tampered detector training complete!")
            print(f"📁 Model saved to: {model_path}")
            print(f"📁 Best model saved to: {best_model_path}")
            print(f"🎯 Best Validation Accuracy: {best_val_acc:.2f}%")
            print(f"📉 Best Validation Loss: {best_val_loss:.4f}")
            print("=" * 50)
            
            return {
                'status': 'completed', 
                'model_path': model_path,
                'best_model_path': str(best_model_path),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'best_val_accuracy': best_val_acc,
                'best_val_loss': best_val_loss
            }
            
        except Exception as e:
            print(f"❌ Tampered detector training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def load_model(self, model_path):
        """Load a trained tampered detector model"""
        if os.path.exists(model_path):
            detector_config = self.config.get('tampered_detection', 
                                             self.config.get('fake_detection', {}))
            
            self.tampered_detector = ModelFactory.create_fake_detector(
                detector_config.get('model', 'resnet18')
            )
            self.tampered_detector.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.tampered_detector.to(self.device)
            return True
        return False

