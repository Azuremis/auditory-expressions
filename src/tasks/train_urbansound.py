"""
Training script for UrbanSound8K classification task.
Implements training loop, evaluation, and logging functionality.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Union, Any
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix as sk_confusion_matrix
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# Local imports
from src.datamodules.urbansound_datamodule import UrbanSoundDataModule
from src.models.cnn_classifier import get_model
from src.utils_audio import spec_augment

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SpecAugmentTransform:
    """
    Apply SpecAugment to spectrogram
    """
    def __init__(
        self, 
        freq_mask_param: int = 10,
        time_mask_param: int = 10,
        freq_mask_count: int = 2,
        time_mask_count: int = 2
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.freq_mask_count = freq_mask_count
        self.time_mask_count = time_mask_count
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return spec_augment(
            x,
            freq_mask_param=self.freq_mask_param,
            time_mask_param=self.time_mask_param,
            freq_mask_count=self.freq_mask_count,
            time_mask_count=self.time_mask_count
        )

def load_config(config_path: str) -> Dict:
    """
    Load configuration from yaml file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Config model_type: {config.get('model_type', 'not specified')}")
    return config

def create_data_loaders(config: Dict, test_fold: int = 1) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders
    
    Args:
        config: Configuration dictionary
        test_fold: Fold to use for testing/validation
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, data_module)
    """
    # Define augmentation transforms
    train_transform = None
    if config.get('augment', {}).get('spec_augment', False):
        train_transform = SpecAugmentTransform(
            freq_mask_param=config.get('augment', {}).get('freq_mask_param', 10),
            time_mask_param=config.get('augment', {}).get('time_mask_param', 10),
            freq_mask_count=config.get('augment', {}).get('freq_mask_count', 2),
            time_mask_count=config.get('augment', {}).get('time_mask_count', 2)
        )
    
    # Use HuggingFace dataset if specified
    use_hf = config.get('use_huggingface', False)
    
    if use_hf:
        from src.datamodules.hf_urbansound_datamodule import HFUrbanSoundDataModule
        logger.info("Using HuggingFace dataset for UrbanSound8K")
        
        # Create data module
        data_module = HFUrbanSoundDataModule(
            test_fold=test_fold,
            batch_size=config.get('batch_size', 32),
            num_workers=config.get('num_workers', 4),
            sample_rate=config.get('sample_rate', 22050),
            n_mels=config.get('n_mels', 128),
            n_fft=config.get('fft_size', 1024),
            hop_length=int(config.get('sample_rate', 22050) * config.get('hop_ms', 25) / 1000),
            win_length=int(config.get('sample_rate', 22050) * config.get('window_ms', 50) / 1000),
            cache_spectrograms=True,
            train_transform=train_transform,
            val_transform=None
        )
    else:
        from src.datamodules.urbansound_datamodule import UrbanSoundDataModule
        logger.info("Using local dataset for UrbanSound8K")
        
        # Create data module
        data_module = UrbanSoundDataModule(
            data_dir=config.get('data_root', './data/urbansound8k'),
            test_fold=test_fold,
            batch_size=config.get('batch_size', 32),
            num_workers=config.get('num_workers', 4),
            sample_rate=config.get('sample_rate', 22050),
            n_mels=config.get('n_mels', 128),
            n_fft=config.get('fft_size', 1024),
            hop_length=int(config.get('sample_rate', 22050) * config.get('hop_ms', 25) / 1000),
            win_length=int(config.get('sample_rate', 22050) * config.get('window_ms', 50) / 1000),
            use_cache=True,
            train_transform=train_transform,
            val_transform=None
        )
    
    # Setup and get dataloaders
    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    
    return train_dataloader, val_dataloader, data_module

def create_model(config: Dict, device: torch.device) -> nn.Module:
    """
    Create model from configuration
    
    Args:
        config: Configuration dictionary
        device: Device to place model on
        
    Returns:
        Initialized model
    """
    logger.info(f"Config keys available: {list(config.keys())}")
    
    model_type = config.get('model_type', 'cnn')
    logger.info(f"Creating model of type: {model_type}")
    
    if model_type == 'cnn':
        model_config = config.get('cnn', {})
        model = get_model(
            model_name=model_config.get('model_name', 'resnet18'),
            num_classes=model_config.get('num_classes', 10),
            pretrained=model_config.get('pretrained', True),
            in_channels=model_config.get('in_channels', 1),
            dropout_rate=model_config.get('dropout', 0.3)
        )
        logger.info(f"Created CNN model: {model_config.get('model_name', 'resnet18')}")
    
    elif model_type == 'transformer':
        from src.models.transformer_audio_classifier import get_transformer_audio_model
        
        model_config = config.get('transformer', {})
        # Calculate approximate sequence length based on audio features
        hop_length = int(config.get('sample_rate', 22050) * config.get('hop_ms', 25) / 1000)
        # Assume average audio is ~4 seconds -> ~400 time frames with hop_length=551
        seq_len = 400  
        
        model = get_transformer_audio_model(
            input_dim=model_config.get('input_dim', config.get('n_mels', 128)),
            seq_len=seq_len,
            conv_channels=model_config.get('conv_channels', 128),
            transformer_dim=model_config.get('transformer_dim', 256),
            nhead=model_config.get('nhead', 8),
            num_encoder_layers=model_config.get('num_encoder_layers', 4),
            dropout=model_config.get('dropout', 0.3),
            num_classes=model_config.get('num_classes', 10),
            input_is_spectrogram=model_config.get('input_is_spectrogram', True)
        )
        logger.info(f"Created Transformer model with {model_config.get('num_encoder_layers', 4)} encoder layers")
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model.to(device)

def create_optimizer(
    model: nn.Module, 
    config: Dict
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """
    Create optimizer and optional LR scheduler
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 3e-4),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Create scheduler if needed
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler_type = config.get('scheduler_type', 'cosine')
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.get('epochs', 30)
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get('scheduler_step_size', 10),
                gamma=config.get('scheduler_gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.get('scheduler_factor', 0.1),
                patience=config.get('scheduler_patience', 5)
            )
    
    return optimizer, scheduler

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Train model for one epoch
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device
        epoch: Current epoch number
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch in progress_bar:
        # Get data
        spectrograms = batch['spectrogram'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(spectrograms)
        logits = outputs['logits']
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / (progress_bar.n + 1),
            'acc': 100.0 * correct / total
        })
    
    # Final metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }

def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on validation set
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # For confusion matrix
    all_preds = []
    all_labels = []
    
    # For top-k accuracy
    k = 3  # Choose top-k value
    topk_correct = 0
    
    # For ROC AUC
    all_probs = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        
        for batch in progress_bar:
            # Get data
            spectrograms = batch['spectrogram'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(spectrograms)
            logits = outputs['logits']
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Calculate top-k accuracy
            _, topk_indices = torch.topk(logits, k, dim=1)
            batch_correct = 0
            for i, label in enumerate(labels):
                if label.item() in topk_indices[i]:
                    batch_correct += 1
            topk_correct += batch_correct
            
            # Store probabilities for ROC AUC
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            
            # Store for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / (progress_bar.n + 1),
                'acc': 100.0 * correct / total
            })
    
    # Final metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    topk_accuracy = 100.0 * topk_correct / total
    
    # Convert lists to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate additional metrics
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    # Kappa score - measures agreement between predicted and true classes
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Balanced accuracy - average of recall for each class
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Calculate ROC AUC score
    try:
        # Get unique classes
        classes = np.unique(y_true)
        # One-hot encode true labels
        y_true_bin = label_binarize(y_true, classes=list(range(len(classes))))
        # Calculate ROC AUC
        roc_auc = roc_auc_score(y_true_bin, all_probs, multi_class='ovr', average='macro')
    except (ValueError, IndexError) as e:
        # In case of issues like some classes not having both positive and negative samples
        logger.warning(f"Could not calculate ROC AUC: {str(e)}")
        roc_auc = np.nan
    
    # Per-class metrics
    class_precision = precision_score(y_true, y_pred, average=None)
    class_recall = recall_score(y_true, y_pred, average=None)
    class_f1 = f1_score(y_true, y_pred, average=None)
    
    # Get detailed classification report as string
    report = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision * 100,  # Convert to percentage to match accuracy scale
        'recall': recall * 100,
        'f1': f1 * 100,
        'kappa': kappa * 100,          # Convert to percentage for consistency
        'balanced_accuracy': balanced_acc * 100,
        'top_k_accuracy': topk_accuracy,
        'roc_auc': roc_auc * 100 if not np.isnan(roc_auc) else np.nan,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'classification_report': report,
        'predictions': y_pred,
        'labels': y_true
    }

def generate_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_map: Dict[int, str],
    output_path: Optional[str] = None,
    normalize: bool = False
) -> np.ndarray:
    """
    Generate and plot confusion matrix
    
    Args:
        labels: Ground truth labels
        predictions: Model predictions
        class_map: Mapping from class ID to class name
        output_path: Path to save the confusion matrix image
        normalize: Whether to normalize the confusion matrix
        
    Returns:
        Confusion matrix as numpy array
    """
    num_classes = len(class_map)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    # Fill confusion matrix
    for true_label, pred_label in zip(labels, predictions):
        conf_matrix[true_label, pred_label] += 1
    
    # Create normalized confusion matrix for visualization
    if normalize:
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        norm_conf_matrix = conf_matrix / row_sums
        plot_matrix = norm_conf_matrix
        fmt = '.2f'
    else:
        plot_matrix = conf_matrix
        fmt = 'd'
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(plot_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.colorbar()
    
    # Add labels
    class_names = [class_map[i] for i in range(num_classes)]
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = plot_matrix.max() / 2.
    for i in range(plot_matrix.shape[0]):
        for j in range(plot_matrix.shape[1]):
            plt.text(j, i, format(plot_matrix[i, j], fmt),
                     ha="center", va="center",
                     color="white" if plot_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Confusion matrix saved to {output_path}")
    
    plt.close()
    
    return conf_matrix

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict,
    output_dir: str,
    name: str = "model"
) -> str:
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Metrics dictionary
        config: Configuration dictionary
        output_dir: Directory to save checkpoint
        name: Name prefix for checkpoint file
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(output_dir, f"{name}_epoch{epoch}.pt")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
        'config': config
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    return checkpoint_path

def train_single_fold(
    config: Dict,
    device: torch.device,
    output_dir: str,
    fold: int,
    wandb_run=None
) -> Dict[str, float]:
    """
    Train and evaluate a model on a single fold
    
    Args:
        config: Configuration dictionary
        device: Device to use for training
        output_dir: Output directory
        fold: Current fold number (1-10)
        wandb_run: WandB run object for logging
        
    Returns:
        Dictionary of validation metrics for the fold
    """
    logger.info(f"train_single_fold - Config model_type: {config.get('model_type', 'not specified')}")
    
    fold_dir = os.path.join(output_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader, data_module = create_data_loaders(config, test_fold=fold)
    logger.info(f"Fold {fold} - Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Fold {fold} - Validation dataset size: {len(val_loader.dataset)}")
    
    # Create model
    model = create_model(config, device)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, config)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_accuracy = 0.0
    best_f1 = 0.0
    best_balanced_acc = 0.0
    num_epochs = config.get('epochs', 30)
    best_metrics = None
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch
        )
        
        # Evaluate
        val_metrics = evaluate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Log metrics
        logger.info(f"Fold {fold} - Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Fold {fold} - Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
        logger.info(f"Fold {fold} - Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
        logger.info(f"Fold {fold} - Val Precision: {val_metrics['precision']:.2f}%, Val Recall: {val_metrics['recall']:.2f}%, Val F1: {val_metrics['f1']:.2f}%")
        logger.info(f"Fold {fold} - Val Balanced Acc: {val_metrics['balanced_accuracy']:.2f}%, Val Kappa: {val_metrics['kappa']:.2f}%, Val Top-3 Acc: {val_metrics['top_k_accuracy']:.2f}%")
        if not np.isnan(val_metrics['roc_auc']):
            logger.info(f"Fold {fold} - Val ROC AUC: {val_metrics['roc_auc']:.2f}%")
        
        # Log to wandb
        if wandb_run is not None:
            log_dict = {
                'fold': fold,
                'epoch': epoch + 1,
                f'fold{fold}_train_loss': train_metrics['loss'],
                f'fold{fold}_train_accuracy': train_metrics['accuracy'],
                f'fold{fold}_val_loss': val_metrics['loss'],
                f'fold{fold}_val_accuracy': val_metrics['accuracy'],
                f'fold{fold}_val_precision': val_metrics['precision'],
                f'fold{fold}_val_recall': val_metrics['recall'],
                f'fold{fold}_val_f1': val_metrics['f1'],
                f'fold{fold}_val_kappa': val_metrics['kappa'],
                f'fold{fold}_val_balanced_accuracy': val_metrics['balanced_accuracy'],
                f'fold{fold}_val_top_k_accuracy': val_metrics['top_k_accuracy'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            if not np.isnan(val_metrics['roc_auc']):
                log_dict[f'fold{fold}_val_roc_auc'] = val_metrics['roc_auc']
                
            wandb_run.log(log_dict)
        
        # Save checkpoint if improved (using multiple criteria)
        if (val_metrics['accuracy'] > best_accuracy or 
            val_metrics['f1'] > best_f1 or 
            val_metrics['balanced_accuracy'] > best_balanced_acc):
                
            if val_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_metrics['accuracy']
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
            if val_metrics['balanced_accuracy'] > best_balanced_acc:
                best_balanced_acc = val_metrics['balanced_accuracy']
                
            best_metrics = val_metrics
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                output_dir=fold_dir,
                name="best_model"
            )
            
            # Generate confusion matrix
            class_map = train_loader.dataset.CLASS_MAPPING
            conf_matrix_path = os.path.join(fold_dir, 'confusion_matrix.png')
            normalized_conf_matrix_path = os.path.join(fold_dir, 'normalized_confusion_matrix.png')
            
            generate_confusion_matrix(
                labels=val_metrics['labels'],
                predictions=val_metrics['predictions'],
                class_map=class_map,
                output_path=conf_matrix_path
            )
            
            # Also generate normalized confusion matrix
            generate_confusion_matrix(
                labels=val_metrics['labels'],
                predictions=val_metrics['predictions'],
                class_map=class_map,
                output_path=normalized_conf_matrix_path,
                normalize=True
            )
            
            # Log confusion matrices to wandb
            if (wandb_run is not None and 
                config.get('wandb', {}).get('log_confusion_matrix', False)):
                wandb_run.log({
                    f"fold{fold}_confusion_matrix": wandb.Image(conf_matrix_path),
                    f"fold{fold}_normalized_confusion_matrix": wandb.Image(normalized_conf_matrix_path)
                })
                
                # Log per-class metrics to wandb
                for i, class_name in class_map.items():
                    wandb_run.log({
                        f"fold{fold}_precision_{class_name}": val_metrics['class_precision'][i] * 100,
                        f"fold{fold}_recall_{class_name}": val_metrics['class_recall'][i] * 100,
                        f"fold{fold}_f1_{class_name}": val_metrics['class_f1'][i] * 100
                    })
    
    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=num_epochs-1,
        metrics=val_metrics,
        config=config,
        output_dir=fold_dir,
        name="final_model"
    )
    
    # Final message
    logger.info(f"Fold {fold} training completed!")
    logger.info(f"Fold {fold} best validation accuracy: {best_accuracy:.2f}%")
    logger.info(f"Fold {fold} best validation F1 score: {best_f1:.2f}%")
    logger.info(f"Fold {fold} best validation balanced accuracy: {best_balanced_acc:.2f}%")
    
    return best_metrics

def main(args):
    """Main training function"""
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or os.path.join(config.get('checkpoints_dir', './checkpoints'), timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Set device
    force_use_gpu = config.get('force_use_gpu', False)
    if force_use_gpu and torch.cuda.is_available():
        device_str = 'cuda'
    else:
        device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")
    
    # Check if we should use cross-validation
    use_cross_validation = config.get('cv', {}).get('cross_validation', False)
    num_folds = config.get('cv', {}).get('folds', 10) if use_cross_validation else 1
    test_fold = config.get('cv', {}).get('fold', 1)
    
    # Initialize wandb if available
    wandb_run = None
    if WANDB_AVAILABLE and config.get('wandb', {}).get('enable', False):
        wandb_config = config.get('wandb', {})
        # Set up wandb with more metadata
        run_name = f"{wandb_config.get('group', 'urbansound')}-{timestamp}"
        
        # Add more context to the run name if in cross-validation mode
        if config.get('cv', {}).get('cross_validation', False):
            run_name += f"-{config.get('cv', {}).get('folds', 10)}fold-cv"
        else:
            run_name += f"-fold{config.get('cv', {}).get('fold', 1)}"
            
        wandb_run = wandb.init(
            project=wandb_config.get('project', 'audio-week5'),
            name=run_name,
            tags=wandb_config.get('tags', []),
            config=config
        )
        
        # Add more custom metrics to be tracked by wandb
        if wandb_run is not None:
            # Define custom x-axis for metrics
            wandb.define_metric("epoch")
            # Set epoch as x-axis for all metrics that have it
            wandb.define_metric("*", step_metric="epoch")
            
            # Create tables for classification metrics in wandb
            if wandb_config.get('detailed_metrics', True):
                # This will be populated during training
                wandb.run.summary["classification_report"] = {}
                
                # Create a table for per-class metrics
                class_metrics_table = wandb.Table(columns=["class", "precision", "recall", "f1", "support"])
                wandb.run.summary["class_metrics"] = class_metrics_table

    if use_cross_validation:
        logger.info(f"Running {num_folds}-fold cross-validation as recommended by dataset creators")
        
        # Storage for metrics across folds
        all_fold_metrics = []
        
        # Train and evaluate on each fold
        for fold in range(1, num_folds + 1):
            logger.info(f"Starting fold {fold}/{num_folds}")
            
            # Train on this fold
            fold_metrics = train_single_fold(
                config=config,
                device=device,
                output_dir=output_dir,
                fold=fold,
                wandb_run=wandb_run
            )
            
            all_fold_metrics.append(fold_metrics)
        
        # Calculate and report average metrics across folds
        avg_accuracy = np.mean([m['accuracy'] for m in all_fold_metrics])
        avg_loss = np.mean([m['loss'] for m in all_fold_metrics])
        avg_precision = np.mean([m['precision'] for m in all_fold_metrics])
        avg_recall = np.mean([m['recall'] for m in all_fold_metrics])
        avg_f1 = np.mean([m['f1'] for m in all_fold_metrics])
        avg_kappa = np.mean([m['kappa'] for m in all_fold_metrics])
        avg_balanced_acc = np.mean([m['balanced_accuracy'] for m in all_fold_metrics])
        avg_top_k = np.mean([m['top_k_accuracy'] for m in all_fold_metrics])
        
        # ROC AUC might be NaN in some folds
        roc_auc_values = [m['roc_auc'] for m in all_fold_metrics if not np.isnan(m['roc_auc'])]
        avg_roc_auc = np.mean(roc_auc_values) if roc_auc_values else np.nan
        
        logger.info("Cross-validation completed!")
        logger.info(f"Average validation accuracy across {num_folds} folds: {avg_accuracy:.2f}%")
        logger.info(f"Average validation precision across {num_folds} folds: {avg_precision:.2f}%")
        logger.info(f"Average validation recall across {num_folds} folds: {avg_recall:.2f}%")
        logger.info(f"Average validation F1 score across {num_folds} folds: {avg_f1:.2f}%")
        logger.info(f"Average validation kappa across {num_folds} folds: {avg_kappa:.2f}%")
        logger.info(f"Average validation balanced accuracy across {num_folds} folds: {avg_balanced_acc:.2f}%")
        logger.info(f"Average validation top-3 accuracy across {num_folds} folds: {avg_top_k:.2f}%")
        if not np.isnan(avg_roc_auc):
            logger.info(f"Average validation ROC AUC across {num_folds} folds: {avg_roc_auc:.2f}%")
        logger.info(f"Average validation loss across {num_folds} folds: {avg_loss:.4f}")
        
        # Log final average metrics to wandb
        if wandb_run is not None:
            log_dict = {
                'final_avg_accuracy': avg_accuracy,
                'final_avg_precision': avg_precision,
                'final_avg_recall': avg_recall,
                'final_avg_f1': avg_f1,
                'final_avg_kappa': avg_kappa,
                'final_avg_balanced_accuracy': avg_balanced_acc,
                'final_avg_top_k_accuracy': avg_top_k,
                'final_avg_loss': avg_loss
            }
            
            if not np.isnan(avg_roc_auc):
                log_dict['final_avg_roc_auc'] = avg_roc_auc
                
            wandb_run.log(log_dict)
            
            # Create a summary of per-fold results
            for fold, metrics in enumerate(all_fold_metrics, 1):
                wandb_run.summary[f"fold{fold}_accuracy"] = metrics['accuracy']
                wandb_run.summary[f"fold{fold}_precision"] = metrics['precision']
                wandb_run.summary[f"fold{fold}_recall"] = metrics['recall']
                wandb_run.summary[f"fold{fold}_f1"] = metrics['f1']
                wandb_run.summary[f"fold{fold}_kappa"] = metrics['kappa']
                wandb_run.summary[f"fold{fold}_balanced_accuracy"] = metrics['balanced_accuracy']
                wandb_run.summary[f"fold{fold}_top_k_accuracy"] = metrics['top_k_accuracy']
                wandb_run.summary[f"fold{fold}_loss"] = metrics['loss']
                
                if not np.isnan(metrics['roc_auc']):
                    wandb_run.summary[f"fold{fold}_roc_auc"] = metrics['roc_auc']
            
            wandb_run.summary["avg_accuracy"] = avg_accuracy
            wandb_run.summary["avg_precision"] = avg_precision
            wandb_run.summary["avg_recall"] = avg_recall
            wandb_run.summary["avg_f1"] = avg_f1
            wandb_run.summary["avg_kappa"] = avg_kappa
            wandb_run.summary["avg_balanced_accuracy"] = avg_balanced_acc
            wandb_run.summary["avg_top_k_accuracy"] = avg_top_k
            wandb_run.summary["avg_loss"] = avg_loss
            
            if not np.isnan(avg_roc_auc):
                wandb_run.summary["avg_roc_auc"] = avg_roc_auc
            
            # Create a classification report summary for wandb
            if wandb_run is not None and config.get('wandb', {}).get('detailed_metrics', True):
                # Aggregate classification reports across folds
                all_reports = [m['classification_report'] for m in all_fold_metrics]
                classes = list(all_reports[0].keys())
                classes = [c for c in classes if c not in ['accuracy', 'macro avg', 'weighted avg']]
                
                # Create a report for each class
                for class_name in classes:
                    precision_values = [report[class_name]['precision'] for report in all_reports]
                    recall_values = [report[class_name]['recall'] for report in all_reports]
                    f1_values = [report[class_name]['f1-score'] for report in all_reports]
                    support_values = [report[class_name]['support'] for report in all_reports]
                    
                    class_metrics_table = wandb.run.summary["class_metrics"]
                    class_metrics_table.add_data(
                        class_name,
                        np.mean(precision_values) * 100,
                        np.mean(recall_values) * 100, 
                        np.mean(f1_values) * 100,
                        np.mean(support_values)
                    )
    else:
        # Single fold training
        logger.info(f"Running single fold training (test fold: {test_fold})")
        logger.info("NOTE: The dataset creators recommend using 10-fold cross-validation instead")
        
        # Create data loaders
        train_loader, val_loader, _ = create_data_loaders(config, test_fold=test_fold)
        logger.info(f"Train dataset size: {len(train_loader.dataset)}")
        logger.info(f"Validation dataset size: {len(val_loader.dataset)}")
        
        # Create model
        model = create_model(config, device)
        
        # Create optimizer and scheduler
        optimizer, scheduler = create_optimizer(model, config)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Initialize wandb model watching
        if wandb_run is not None:
            wandb.watch(model)
        
        # Training loop
        best_accuracy = 0.0
        best_f1 = 0.0
        best_balanced_acc = 0.0
        num_epochs = config.get('epochs', 30)
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch
            )
            
            # Evaluate
            val_metrics = evaluate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device
            )
            
            # Update learning rate
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            logger.info(f"Val Precision: {val_metrics['precision']:.2f}%, Val Recall: {val_metrics['recall']:.2f}%, Val F1: {val_metrics['f1']:.2f}%")
            logger.info(f"Val Balanced Acc: {val_metrics['balanced_accuracy']:.2f}%, Val Kappa: {val_metrics['kappa']:.2f}%, Val Top-3 Acc: {val_metrics['top_k_accuracy']:.2f}%")
            if not np.isnan(val_metrics['roc_auc']):
                logger.info(f"Val ROC AUC: {val_metrics['roc_auc']:.2f}%")
            
            # Log to wandb
            if wandb_run is not None:
                log_dict = {
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'val_f1': val_metrics['f1'],
                    'val_kappa': val_metrics['kappa'],
                    'val_balanced_accuracy': val_metrics['balanced_accuracy'],
                    'val_top_k_accuracy': val_metrics['top_k_accuracy'],
                    'learning_rate': optimizer.param_groups[0]['lr']
                }
                
                if not np.isnan(val_metrics['roc_auc']):
                    log_dict['val_roc_auc'] = val_metrics['roc_auc']
                    
                wandb_run.log(log_dict)
            
            # Save checkpoint if improved (using multiple criteria)
            if (val_metrics['accuracy'] > best_accuracy or 
                val_metrics['f1'] > best_f1 or 
                val_metrics['balanced_accuracy'] > best_balanced_acc):
                    
                if val_metrics['accuracy'] > best_accuracy:
                    best_accuracy = val_metrics['accuracy']
                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                if val_metrics['balanced_accuracy'] > best_balanced_acc:
                    best_balanced_acc = val_metrics['balanced_accuracy']
                    
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    config=config,
                    output_dir=output_dir,
                    name="best_model"
                )
                
                # Generate confusion matrix
                class_map = train_loader.dataset.CLASS_MAPPING
                conf_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
                normalized_conf_matrix_path = os.path.join(output_dir, 'normalized_confusion_matrix.png')
                
                generate_confusion_matrix(
                    labels=val_metrics['labels'],
                    predictions=val_metrics['predictions'],
                    class_map=class_map,
                    output_path=conf_matrix_path
                )
                
                # Also generate normalized confusion matrix
                generate_confusion_matrix(
                    labels=val_metrics['labels'],
                    predictions=val_metrics['predictions'],
                    class_map=class_map,
                    output_path=normalized_conf_matrix_path,
                    normalize=True
                )
                
                # Log confusion matrices to wandb
                if (wandb_run is not None and 
                    config.get('wandb', {}).get('log_confusion_matrix', False)):
                    wandb_run.log({
                        "confusion_matrix": wandb.Image(conf_matrix_path),
                        "normalized_confusion_matrix": wandb.Image(normalized_conf_matrix_path)
                    })
                    
                    # Add per-class metrics to wandb
                    if config.get('wandb', {}).get('detailed_metrics', True):
                        for i, class_name in class_map.items():
                            wandb_run.log({
                                f"precision_{class_name}": val_metrics['class_precision'][i] * 100,
                                f"recall_{class_name}": val_metrics['class_recall'][i] * 100,
                                f"f1_{class_name}": val_metrics['class_f1'][i] * 100
                            })
                        
                        # Create a table for per-class metrics in the summary
                        class_metrics_table = wandb.run.summary["class_metrics"]
                        report = val_metrics['classification_report']
                        for class_name in class_map.values():
                            if class_name in report:
                                class_data = report[class_name]
                                class_metrics_table.add_data(
                                    class_name,
                                    class_data['precision'] * 100,
                                    class_data['recall'] * 100,
                                    class_data['f1-score'] * 100,
                                    class_data['support']
                                )
        
        # Save final model
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=num_epochs-1,
            metrics=val_metrics,
            config=config,
            output_dir=output_dir,
            name="final_model"
        )
        
        # Final message
        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {best_accuracy:.2f}%")
        logger.info(f"Best validation F1 score: {best_f1:.2f}%")
        logger.info(f"Best validation balanced accuracy: {best_balanced_acc:.2f}%")
    
    logger.info(f"Model checkpoints saved to {output_dir}")
    
    # Close wandb
    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UrbanSound8K classifier")
    parser.add_argument('--config', type=str, default='configs/urbansound_cnn.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save outputs (default: checkpoints_dir from config)')
    
    args = parser.parse_args()
    main(args) 