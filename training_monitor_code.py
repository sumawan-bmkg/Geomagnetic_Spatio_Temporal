
# Training Monitoring Code - Add to training loop

import torch
import psutil
import GPUtil

class TrainingMonitor:
    def __init__(self, config):
        self.config = config
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.lr_patience_counter = 0
        
    def log_epoch_metrics(self, epoch, train_loss, val_loss, metrics, model, optimizer):
        # Memory monitoring
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_pct = gpu_memory / torch.cuda.get_device_properties(0).total_memory * 1024**3 * 100
            
            if gpu_memory_pct > self.config['memory_monitoring']['memory_warning_threshold'] * 100:
                logger.warning(f"High GPU memory usage: {gpu_memory_pct:.1f}%")
        
        # Loss monitoring
        if val_loss < self.best_val_loss - self.config['loss_monitoring']['min_delta']:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'best_model_epoch_{epoch}.pth')
        else:
            self.patience_counter += 1
            
        # Learning rate reduction
        if self.patience_counter >= self.config['loss_monitoring']['lr_reduction_patience']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.config['loss_monitoring']['lr_reduction_factor']
            logger.info(f"Reduced learning rate to {param_group['lr']:.2e}")
            self.lr_patience_counter = 0
        
        # Early stopping
        if self.patience_counter >= self.config['loss_monitoring']['early_stopping_patience']:
            logger.info("Early stopping triggered")
            return True
        
        return False
