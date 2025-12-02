"""
Main training loop and trainer class.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    
    # Basic settings
    epochs: int = 30
    batch_size: int = 4
    num_workers: int = 4
    
    # Optimizer
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    
    # Sampling
    num_surface_samples: int = 4096
    num_freespace_samples: int = 4096
    freespace_jitter_min: float = 0.05
    freespace_jitter_max: float = 0.5
    
    # Loss weights
    surface_weight: float = 1.0
    freespace_weight: float = 0.5
    eikonal_weight: float = 0.1
    planar_weight: float = 0.3
    normal_weight: float = 0.2
    manhattan_weight: float = 0.1
    
    # Training options
    use_amp: bool = True
    grad_clip: float = 1.0
    
    # Logging
    log_every: int = 100
    val_every: int = 1
    save_every: int = 5
    
    # Output
    output_dir: str = "output"
    experiment_name: str = "neural_sdf"


class Trainer:
    """
    Main trainer class for neural 3D reconstruction.
    
    Handles training loop, validation, checkpointing, and logging.
    
    Args:
        model: Neural SDF model
        train_loader: Training data loader
        val_loader: Optional validation data loader
        config: Training configuration
        device: Device to train on
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config or TrainingConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if self.config.use_amp else None
        
        # Logging
        self.writer = None
        if HAS_TENSORBOARD:
            log_dir = os.path.join(self.config.output_dir, 'logs', self.config.experiment_name)
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            
        # Checkpointing
        self.checkpoint_dir = os.path.join(
            self.config.output_dir, 'checkpoints', self.config.experiment_name
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
            
    def _create_scheduler(self):
        """Create learning rate scheduler based on config."""
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )
        elif self.config.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        else:
            return None
            
    def train(self) -> Dict[str, float]:
        """
        Run full training loop.
        
        Returns:
            Dictionary of final metrics
        """
        print(f"Starting training for {self.config.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            # Training epoch
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = {}
            if self.val_loader is not None and (epoch + 1) % self.config.val_every == 0:
                val_metrics = self.validate()
                
            # Logging
            self._log_epoch(train_metrics, val_metrics)
            
            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    self.scheduler.step()
                    
            # Checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint()
                
            # Best model
            val_loss = val_metrics.get('loss', train_metrics['loss'])
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
                
        # Final save
        self.save_checkpoint('final_model.pt')
        
        if self.writer:
            self.writer.close()
            
        return {'best_val_loss': self.best_val_loss}
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._to_device(batch)
            
            # Forward pass
            loss, components = self.train_step(batch)
            
            # Accumulate metrics
            total_loss += loss.item()
            for key, value in components.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Logging
            if (self.global_step + 1) % self.config.log_every == 0:
                self._log_step(loss.item(), components)
                
            self.global_step += 1
            
        # Average metrics
        avg_metrics = {'loss': total_loss / num_batches}
        for key, value in loss_components.items():
            avg_metrics[key] = value / num_batches
            
        return avg_metrics
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """
        Single training step.
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            loss: Total loss
            components: Dictionary of loss components
        """
        self.optimizer.zero_grad()
        
        # Extract data from batch
        depth = batch['depth']  # (B, H, W)
        pose = batch['pose']    # (B, 4, 4)
        K = batch['K']          # (B, 3, 3)
        valid_mask = batch.get('valid_mask', depth > 0)
        
        B, H, W = depth.shape
        
        # Sample surface points
        surface_points, camera_centers = self._sample_surface_points(
            depth, pose, K, valid_mask
        )
        
        # Sample freespace points
        freespace_points, freespace_targets = self._sample_freespace_points(
            surface_points, camera_centers
        )
        
        # Forward pass with mixed precision
        with autocast(enabled=self.config.use_amp):
            # Surface points
            surface_out = self.model(surface_points)
            sdf_surface = surface_out['sdf']
            
            # Freespace points
            freespace_out = self.model(freespace_points)
            sdf_freespace = freespace_out['sdf']
            
            # Compute gradients for Eikonal loss
            all_points = torch.cat([surface_points, freespace_points], dim=1)
            all_points = all_points.requires_grad_(True)
            all_out = self.model(all_points)
            all_sdf = all_out['sdf']
            
            gradients = torch.autograd.grad(
                outputs=all_sdf,
                inputs=all_points,
                grad_outputs=torch.ones_like(all_sdf),
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Compute losses
            loss_surface = torch.abs(sdf_surface).mean()
            loss_freespace = torch.abs(
                sdf_freespace.squeeze(-1) - freespace_targets
            ).mean()
            
            grad_norm = torch.norm(gradients, dim=-1)
            loss_eikonal = ((grad_norm - 1.0) ** 2).mean()
            
            # Total loss
            loss = (
                self.config.surface_weight * loss_surface +
                self.config.freespace_weight * loss_freespace +
                self.config.eikonal_weight * loss_eikonal
            )
            
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                
            self.optimizer.step()
            
        components = {
            'surface': loss_surface.item(),
            'freespace': loss_freespace.item(),
            'eikonal': loss_eikonal.item(),
        }
        
        return loss, components
        
    def _sample_surface_points(
        self,
        depth: torch.Tensor,
        pose: torch.Tensor,
        K: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> tuple:
        """Sample 3D surface points from depth maps."""
        B, H, W = depth.shape
        device = depth.device
        N_samples = self.config.num_surface_samples
        
        # Flatten for sampling
        depth_flat = depth.view(B, -1)
        valid_flat = valid_mask.view(B, -1)
        
        all_surface_points = []
        all_camera_centers = []
        
        for b in range(B):
            # Get valid indices
            valid_idx = torch.where(valid_flat[b])[0]
            
            if len(valid_idx) < N_samples:
                # Pad with repetition
                idx = valid_idx[torch.randint(len(valid_idx), (N_samples,))]
            else:
                # Random sample
                perm = torch.randperm(len(valid_idx))[:N_samples]
                idx = valid_idx[perm]
                
            # Get pixel coordinates
            v = idx // W
            u = idx % W
            z = depth_flat[b, idx]
            
            # Back-project to camera coordinates
            fx, fy = K[b, 0, 0], K[b, 1, 1]
            cx, cy = K[b, 0, 2], K[b, 1, 2]
            
            x = (u.float() - cx) * z / fx
            y = (v.float() - cy) * z / fy
            
            pts_cam = torch.stack([x, y, z], dim=-1)  # (N, 3)
            
            # Transform to world coordinates
            R = pose[b, :3, :3]
            t = pose[b, :3, 3]
            
            pts_world = pts_cam @ R.T + t
            
            all_surface_points.append(pts_world)
            all_camera_centers.append(t.expand(N_samples, -1))
            
        surface_points = torch.stack(all_surface_points, dim=0)  # (B, N, 3)
        camera_centers = torch.stack(all_camera_centers, dim=0)  # (B, N, 3)
        
        return surface_points, camera_centers
        
    def _sample_freespace_points(
        self,
        surface_points: torch.Tensor,
        camera_centers: torch.Tensor
    ) -> tuple:
        """Sample freespace points between camera and surface."""
        # Ray direction (from surface to camera)
        ray_dirs = camera_centers - surface_points
        ray_lengths = torch.norm(ray_dirs, dim=-1, keepdim=True)
        ray_dirs = ray_dirs / (ray_lengths + 1e-8)
        
        # Random jitter distance
        jitter = torch.rand_like(ray_lengths)
        jitter = jitter * (self.config.freespace_jitter_max - self.config.freespace_jitter_min)
        jitter = jitter + self.config.freespace_jitter_min
        
        # Clamp to not go past camera
        jitter = torch.min(jitter, ray_lengths * 0.9)
        
        # Sample freespace points
        freespace_points = surface_points + ray_dirs * jitter
        
        # Target SDF values (distance to surface)
        freespace_targets = jitter.squeeze(-1)
        
        return freespace_points, freespace_targets
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = self._to_device(batch)
            
            # Forward pass
            depth = batch['depth']
            pose = batch['pose']
            K = batch['K']
            valid_mask = batch.get('valid_mask', depth > 0)
            
            surface_points, camera_centers = self._sample_surface_points(
                depth, pose, K, valid_mask
            )
            
            surface_out = self.model(surface_points)
            sdf_surface = surface_out['sdf']
            
            loss = torch.abs(sdf_surface).mean()
            total_loss += loss.item()
            num_batches += 1
            
        return {'loss': total_loss / num_batches}
        
    def _to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
    def _log_step(self, loss: float, components: Dict[str, float]):
        """Log training step metrics."""
        if self.writer is None:
            return
            
        self.writer.add_scalar('train/loss', loss, self.global_step)
        for key, value in components.items():
            self.writer.add_scalar(f'train/{key}', value, self.global_step)
            
        self.writer.add_scalar(
            'train/lr',
            self.optimizer.param_groups[0]['lr'],
            self.global_step
        )
        
    def _log_epoch(self, train_metrics: Dict, val_metrics: Dict):
        """Log epoch metrics."""
        print(f"\nEpoch {self.current_epoch} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.6f}")
        
        for key, value in train_metrics.items():
            if key != 'loss':
                print(f"  Train {key}: {value:.6f}")
                
        if val_metrics:
            print(f"  Val Loss: {val_metrics['loss']:.6f}")
            
        if self.writer:
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], self.current_epoch)
            if val_metrics:
                self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], self.current_epoch)
                
    def save_checkpoint(self, filename: Optional[str] = None):
        """Save training checkpoint."""
        if filename is None:
            filename = f'checkpoint_epoch{self.current_epoch}.pt'
            
        path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        print(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
