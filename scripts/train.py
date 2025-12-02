#!/usr/bin/env python3
"""
Training script for neural 3D reconstruction.

Usage:
    python scripts/train.py --config config/experiment/cse_warehouse.yaml
    python scripts/train.py --config config/default.yaml --data_root data/warehouse_extracted
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import yaml
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CSEDataset, CSEMultiCameraDataset
from src.models import NeuralSDF, NeuralSDFWithPlanar, HashGridEncoding
from src.losses import SDFLoss, PlanarConsistencyLoss, ManhattanLoss
from src.training import Trainer, TrainingConfig, CheckpointManager, get_scheduler


def setup_logging(output_dir: Path, name: str = 'train'):
    """Configure logging to file and console."""
    log_file = output_dir / f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, device: str = 'cuda') -> torch.nn.Module:
    """Create neural SDF model from config."""
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'neural_sdf_planar')
    
    # Create encoding
    encoding_config = model_config.get('encoding', {})
    encoding_type = encoding_config.get('type', 'hashgrid')
    
    if encoding_type == 'hashgrid':
        encoding = HashGridEncoding(
            n_levels=encoding_config.get('n_levels', 16),
            n_features_per_level=encoding_config.get('n_features_per_level', 2),
            log2_hashmap_size=encoding_config.get('log2_hashmap_size', 19),
            base_resolution=encoding_config.get('base_resolution', 16),
            finest_resolution=encoding_config.get('finest_resolution', 512)
        )
        input_dim = encoding.output_dim
    else:
        encoding = None
        input_dim = 3
    
    # Create model
    if model_type == 'neural_sdf':
        model = NeuralSDF(
            in_features=input_dim,
            hidden_features=model_config.get('hidden_dim', 256),
            hidden_layers=model_config.get('num_layers', 4),
            out_features=1,
            encoding=encoding
        )
    elif model_type == 'neural_sdf_planar':
        model = NeuralSDFWithPlanar(
            in_features=input_dim,
            hidden_features=model_config.get('hidden_dim', 256),
            hidden_layers=model_config.get('num_layers', 4),
            num_planes=model_config.get('num_planes', 32),
            encoding=encoding
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def create_loss_function(config: dict) -> torch.nn.Module:
    """Create combined loss function from config."""
    loss_config = config.get('loss', {})
    
    # Base SDF loss
    sdf_loss = SDFLoss(
        surface_weight=loss_config.get('surface_weight', 1.0),
        freespace_weight=loss_config.get('freespace_weight', 0.1),
        eikonal_weight=loss_config.get('eikonal_weight', 0.1)
    )
    
    # Additional losses
    losses = {'sdf': (sdf_loss, 1.0)}
    
    if loss_config.get('use_planar', True):
        planar_weight = loss_config.get('planar_weight', 0.1)
        losses['planar'] = (PlanarConsistencyLoss(), planar_weight)
        
    if loss_config.get('use_manhattan', False):
        manhattan_weight = loss_config.get('manhattan_weight', 0.05)
        losses['manhattan'] = (ManhattanLoss(), manhattan_weight)
    
    return losses


def create_dataloader(config: dict, split: str = 'train') -> torch.utils.data.DataLoader:
    """Create dataloader from config."""
    data_config = config.get('data', {})
    
    # Dataset
    data_root = Path(data_config.get('data_root', 'data/warehouse_extracted'))
    sequences = data_config.get(f'{split}_sequences', None)
    
    if sequences is None:
        # Auto-detect sequences
        sequences = [d.name for d in data_root.iterdir() if d.is_dir()]
        
    # Create dataset
    multi_camera = data_config.get('multi_camera', False)
    
    if multi_camera:
        dataset = CSEMultiCameraDataset(
            root_dir=str(data_root),
            sequences=sequences,
            max_frames=data_config.get('max_frames', 1000)
        )
    else:
        dataset = CSEDataset(
            root_dir=str(data_root),
            sequences=sequences,
            cameras=['left', 'right'],
            max_frames=data_config.get('max_frames', 1000)
        )
    
    # Create dataloader
    batch_size = data_config.get('batch_size', 4) if split == 'train' else 1
    shuffle = (split == 'train')
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


def main():
    parser = argparse.ArgumentParser(description='Train neural 3D reconstruction')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    exp_name = config.get('experiment_name', 'default')
    output_dir = Path(args.output) / exp_name / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting training experiment: {exp_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config, device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss functions
    logger.info("Creating loss functions...")
    losses = create_loss_function(config)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader = create_dataloader(config, 'train')
    val_loader = create_dataloader(config, 'val')
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Create optimizer
    train_config = config.get('training', {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.get('learning_rate', 1e-4),
        weight_decay=train_config.get('weight_decay', 1e-5)
    )
    
    # Create scheduler
    scheduler = get_scheduler(
        optimizer,
        scheduler_type=train_config.get('scheduler', 'cosine'),
        warmup_epochs=train_config.get('warmup_epochs', 5),
        total_epochs=train_config.get('num_epochs', 100),
        min_lr=train_config.get('min_lr', 1e-6)
    )
    
    # Create checkpoint manager
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=output_dir / 'checkpoints',
        max_checkpoints=5,
        metric_name='val_loss',
        mode='min'
    )
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = checkpoint_mgr.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
    
    # Create training config
    training_config = TrainingConfig(
        num_epochs=train_config.get('num_epochs', 100),
        learning_rate=train_config.get('learning_rate', 1e-4),
        gradient_clip=train_config.get('gradient_clip', 1.0),
        log_every=train_config.get('log_every', 100),
        val_every=train_config.get('val_every', 1),
        save_every=train_config.get('save_every', 5),
        use_amp=train_config.get('use_amp', True),
        device=device
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=training_config,
        checkpoint_manager=checkpoint_mgr
    )
    
    # Define forward pass for SDF training
    def compute_loss(batch, model):
        """Compute combined loss for a batch."""
        points = batch['points'].to(device)
        sdf_gt = batch['sdf'].to(device)
        
        # Forward pass
        output = model(points)
        
        # Compute losses
        total_loss = 0
        loss_dict = {}
        
        for name, (loss_fn, weight) in losses.items():
            if name == 'sdf':
                loss = loss_fn(output, sdf_gt, points)
            else:
                loss = loss_fn(output, points)
            loss_dict[name] = loss.item()
            total_loss += weight * loss
            
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    # Training loop
    logger.info("Starting training...")
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, training_config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{training_config.num_epochs}")
        
        # Train
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            loss, loss_dict = compute_loss(batch, model)
            
            optimizer.zero_grad()
            loss.backward()
            
            if training_config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    training_config.gradient_clip
                )
                
            optimizer.step()
            
            train_losses.append(loss_dict['total'])
            
            if (batch_idx + 1) % training_config.log_every == 0:
                logger.info(
                    f"  Batch {batch_idx + 1}/{len(train_loader)} - "
                    f"Loss: {loss_dict['total']:.6f}"
                )
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        logger.info(f"  Train Loss: {avg_train_loss:.6f}")
        
        # Validate
        if (epoch + 1) % training_config.val_every == 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    _, loss_dict = compute_loss(batch, model)
                    val_losses.append(loss_dict['total'])
                    
            avg_val_loss = sum(val_losses) / len(val_losses)
            logger.info(f"  Val Loss: {avg_val_loss:.6f}")
            
            # Save checkpoint
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                
            checkpoint_mgr.save(
                state_dict={
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'config': config
                },
                epoch=epoch,
                step=epoch * len(train_loader),
                metrics={'val_loss': avg_val_loss, 'train_loss': avg_train_loss},
                is_best=is_best
            )
        
        # Step scheduler
        if scheduler:
            scheduler.step()
            
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Checkpoints saved to: {output_dir / 'checkpoints'}")


if __name__ == '__main__':
    main()
