#!/usr/bin/env python3
"""
Training Health Monitor - Real-time monitoring untuk maraton training 2-4 jam
Memantau "kesehatan model" dan memberikan alert untuk berbagai kondisi

Author: Kiro AI Assistant  
Date: April 16, 2026
"""
import os
import time
import psutil
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_health_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingHealthMonitor:
    """
    Monitor kesehatan training secara real-time
    """
    
    def __init__(self, log_file_path="final_production_training.log"):
        self.log_file_path = log_file_path
        self.start_time = datetime.now()
        self.health_history = {
            'timestamps': [],
            'losses': [],
            'cpu_usage': [],
            'memory_usage': [],
            'stages': [],
            'epochs': [],
            'learning_rates': []
        }
        
        # Health thresholds
        self.thresholds = {
            'loss_plateau_epochs': 3,  # Alert jika loss datar 3 epoch
            'min_cpu_usage': 30,       # Alert jika CPU < 30%
            'max_memory_usage': 90,    # Alert jika memory > 90%
            'loss_explosion_factor': 10,  # Alert jika loss naik 10x
            'min_loss_decrease': 0.01  # Minimum penurunan loss yang diharapkan
        }
        
        logger.info("=== TRAINING HEALTH MONITOR INITIALIZED ===")
        logger.info(f"Monitoring log file: {log_file_path}")
        logger.info("Monitoring indicators:")
        logger.info("  - Loss convergence patterns")
        logger.info("  - Stage transition health")
        logger.info("  - CPU/Memory utilization")
        logger.info("  - Learning rate dynamics")
    
    def parse_training_log(self):
        """Parse log file untuk extract training metrics"""
        if not os.path.exists(self.log_file_path):
            return None
        
        try:
            with open(self.log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            current_metrics = {
                'current_stage': None,
                'current_epoch': None,
                'latest_loss': None,
                'latest_cpu': None,
                'latest_memory': None,
                'latest_lr': None,
                'total_epochs_completed': 0
            }
            
            for line in lines[-50:]:  # Check last 50 lines
                # Extract stage info
                if "STARTING STAGE" in line:
                    try:
                        stage_num = int(line.split("STAGE ")[1].split(":")[0])
                        current_metrics['current_stage'] = stage_num
                    except:
                        pass
                
                # Extract epoch info
                if "Stage" in line and "Epoch" in line and "/" in line:
                    try:
                        parts = line.split("Epoch ")[1].split("/")
                        current_epoch = int(parts[0])
                        current_metrics['current_epoch'] = current_epoch
                    except:
                        pass
                
                # Extract loss info
                if "Loss:" in line:
                    try:
                        loss_str = line.split("Loss: ")[1].split()[0]
                        current_metrics['latest_loss'] = float(loss_str)
                    except:
                        pass
                
                # Extract CPU usage
                if "CPU Usage:" in line:
                    try:
                        cpu_str = line.split("CPU Usage: ")[1].split("%")[0]
                        current_metrics['latest_cpu'] = float(cpu_str)
                    except:
                        pass
                
                # Extract Memory usage
                if "Memory Usage:" in line:
                    try:
                        mem_str = line.split("Memory Usage: ")[1].split("%")[0]
                        current_metrics['latest_memory'] = float(mem_str)
                    except:
                        pass
                
                # Extract Learning Rate
                if "Learning Rate:" in line:
                    try:
                        lr_str = line.split("Learning Rate: ")[1].split()[0]
                        current_metrics['latest_lr'] = float(lr_str)
                    except:
                        pass
            
            return current_metrics
            
        except Exception as e:
            logger.error(f"Error parsing log: {e}")
            return None
    
    def check_loss_health(self, current_loss, stage, epoch):
        """Check kesehatan loss patterns"""
        alerts = []
        
        if current_loss is None:
            return ["Loss value not found in logs"]
        
        # Check for suspiciously low loss
        if current_loss < 0.001:
            alerts.append(f"⚠️  LOSS SUSPICIOUSLY LOW: {current_loss:.6f} - Possible gradient vanishing")
        
        # Check for loss explosion
        if len(self.health_history['losses']) > 0:
            prev_loss = self.health_history['losses'][-1]
            if current_loss > prev_loss * self.thresholds['loss_explosion_factor']:
                alerts.append(f"🚨 LOSS EXPLOSION: {prev_loss:.4f} → {current_loss:.4f}")
        
        # Check for loss plateau (Stage 1 specific)
        if stage == 1 and epoch >= 5:
            recent_losses = self.health_history['losses'][-5:] if len(self.health_history['losses']) >= 5 else []
            if len(recent_losses) >= 3:
                loss_variance = np.var(recent_losses)
                if loss_variance < 0.0001:
                    alerts.append(f"⚠️  LOSS PLATEAU DETECTED: Variance={loss_variance:.6f} - Consider increasing LR")
        
        # Expected loss ranges per stage
        expected_ranges = {
            1: (0.1, 2.0),   # Binary classification
            2: (0.5, 3.0),   # Magnitude + Binary
            3: (0.3, 2.5)    # Full multi-task
        }
        
        if stage in expected_ranges:
            min_loss, max_loss = expected_ranges[stage]
            if not (min_loss <= current_loss <= max_loss):
                alerts.append(f"⚠️  LOSS OUT OF EXPECTED RANGE: {current_loss:.4f} (expected: {min_loss}-{max_loss})")
        
        return alerts
    
    def check_stage_transition_health(self, current_stage, prev_stage, current_loss, prev_loss):
        """Check kesehatan transisi antar stage"""
        alerts = []
        
        if current_stage != prev_stage and prev_stage is not None:
            logger.info(f"🔄 STAGE TRANSITION DETECTED: {prev_stage} → {current_stage}")
            
            # Wajar jika loss naik saat transisi stage
            if current_loss and prev_loss and current_loss > prev_loss:
                loss_increase = (current_loss - prev_loss) / prev_loss * 100
                if loss_increase > 50:
                    alerts.append(f"📈 NORMAL STAGE TRANSITION: Loss increased {loss_increase:.1f}% (expected)")
                else:
                    alerts.append(f"✅ SMOOTH STAGE TRANSITION: Loss increased {loss_increase:.1f}%")
            
            # Alert jika loss turun drastis (tidak normal)
            elif current_loss and prev_loss and current_loss < prev_loss * 0.5:
                alerts.append(f"⚠️  UNUSUAL STAGE TRANSITION: Loss dropped too much")
        
        return alerts
    
    def check_system_health(self, cpu_usage, memory_usage):
        """Check kesehatan sistem (CPU/Memory)"""
        alerts = []
        
        # CPU Health Check
        if cpu_usage is not None:
            if cpu_usage < self.thresholds['min_cpu_usage']:
                alerts.append(f"⚠️  LOW CPU USAGE: {cpu_usage:.1f}% - Backpropagation might be slow")
            elif cpu_usage > 80:
                alerts.append(f"🔥 HIGH CPU USAGE: {cpu_usage:.1f}% - Model working hard! (Good sign)")
            else:
                alerts.append(f"✅ HEALTHY CPU USAGE: {cpu_usage:.1f}%")
        
        # Memory Health Check  
        if memory_usage is not None:
            if memory_usage > self.thresholds['max_memory_usage']:
                alerts.append(f"🚨 HIGH MEMORY USAGE: {memory_usage:.1f}% - Risk of OOM")
            elif memory_usage > 70:
                alerts.append(f"⚠️  MODERATE MEMORY USAGE: {memory_usage:.1f}%")
            else:
                alerts.append(f"✅ HEALTHY MEMORY USAGE: {memory_usage:.1f}%")
        
        return alerts
    
    def check_learning_rate_health(self, current_lr, stage, epoch):
        """Check kesehatan learning rate"""
        alerts = []
        
        if current_lr is None:
            return ["Learning rate not found in logs"]
        
        # Expected LR ranges
        if current_lr < 1e-6:
            alerts.append(f"⚠️  LEARNING RATE TOO LOW: {current_lr:.2e} - Training might be too slow")
        elif current_lr > 0.1:
            alerts.append(f"⚠️  LEARNING RATE TOO HIGH: {current_lr:.2e} - Risk of instability")
        else:
            alerts.append(f"✅ HEALTHY LEARNING RATE: {current_lr:.2e}")
        
        return alerts
    
    def estimate_completion_time(self, current_stage, current_epoch):
        """Estimasi waktu selesai berdasarkan progress"""
        if current_stage is None or current_epoch is None:
            return "Unknown"
        
        # Total epochs: 25 per stage × 3 stages = 75
        total_epochs = 75
        completed_epochs = (current_stage - 1) * 25 + current_epoch
        
        elapsed_time = datetime.now() - self.start_time
        if completed_epochs > 0:
            time_per_epoch = elapsed_time / completed_epochs
            remaining_epochs = total_epochs - completed_epochs
            estimated_remaining = time_per_epoch * remaining_epochs
            
            completion_time = datetime.now() + estimated_remaining
            return f"{estimated_remaining} (ETA: {completion_time.strftime('%H:%M:%S')})"
        
        return "Calculating..."
    
    def generate_health_report(self):
        """Generate comprehensive health report"""
        metrics = self.parse_training_log()
        
        if not metrics:
            logger.warning("Could not parse training metrics")
            return
        
        # Update history
        now = datetime.now()
        self.health_history['timestamps'].append(now)
        
        if metrics['latest_loss'] is not None:
            self.health_history['losses'].append(metrics['latest_loss'])
        if metrics['latest_cpu'] is not None:
            self.health_history['cpu_usage'].append(metrics['latest_cpu'])
        if metrics['latest_memory'] is not None:
            self.health_history['memory_usage'].append(metrics['latest_memory'])
        if metrics['current_stage'] is not None:
            self.health_history['stages'].append(metrics['current_stage'])
        if metrics['current_epoch'] is not None:
            self.health_history['epochs'].append(metrics['current_epoch'])
        if metrics['latest_lr'] is not None:
            self.health_history['learning_rates'].append(metrics['latest_lr'])
        
        # Generate alerts
        all_alerts = []
        
        # Loss health
        prev_stage = self.health_history['stages'][-2] if len(self.health_history['stages']) >= 2 else None
        prev_loss = self.health_history['losses'][-2] if len(self.health_history['losses']) >= 2 else None
        
        loss_alerts = self.check_loss_health(
            metrics['latest_loss'], 
            metrics['current_stage'], 
            metrics['current_epoch']
        )
        all_alerts.extend(loss_alerts)
        
        # Stage transition health
        transition_alerts = self.check_stage_transition_health(
            metrics['current_stage'], 
            prev_stage,
            metrics['latest_loss'], 
            prev_loss
        )
        all_alerts.extend(transition_alerts)
        
        # System health
        system_alerts = self.check_system_health(
            metrics['latest_cpu'], 
            metrics['latest_memory']
        )
        all_alerts.extend(system_alerts)
        
        # Learning rate health
        lr_alerts = self.check_learning_rate_health(
            metrics['latest_lr'], 
            metrics['current_stage'], 
            metrics['current_epoch']
        )
        all_alerts.extend(lr_alerts)
        
        # Generate report
        elapsed_time = datetime.now() - self.start_time
        eta = self.estimate_completion_time(metrics['current_stage'], metrics['current_epoch'])
        
        logger.info("=" * 80)
        logger.info("🏥 TRAINING HEALTH REPORT")
        logger.info("=" * 80)
        logger.info(f"⏱️  Elapsed Time: {elapsed_time}")
        logger.info(f"🎯 Current Progress: Stage {metrics['current_stage']}, Epoch {metrics['current_epoch']}")
        logger.info(f"📊 Latest Loss: {metrics['latest_loss']:.6f}" if metrics['latest_loss'] else "📊 Latest Loss: N/A")
        logger.info(f"💻 CPU Usage: {metrics['latest_cpu']:.1f}%" if metrics['latest_cpu'] else "💻 CPU Usage: N/A")
        logger.info(f"🧠 Memory Usage: {metrics['latest_memory']:.1f}%" if metrics['latest_memory'] else "🧠 Memory Usage: N/A")
        logger.info(f"📈 Learning Rate: {metrics['latest_lr']:.2e}" if metrics['latest_lr'] else "📈 Learning Rate: N/A")
        logger.info(f"⏰ ETA: {eta}")
        
        logger.info("\n🚨 HEALTH ALERTS:")
        for alert in all_alerts:
            logger.info(f"   {alert}")
        
        if not all_alerts:
            logger.info("   ✅ All systems healthy!")
        
        logger.info("=" * 80)
        
        return metrics, all_alerts
    
    def plot_training_curves(self):
        """Plot real-time training curves"""
        if len(self.health_history['losses']) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Real-time Training Health Monitor', fontsize=16)
        
        # Loss curve
        if self.health_history['losses']:
            axes[0, 0].plot(self.health_history['losses'], 'b-', marker='o')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Monitoring Points')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # CPU usage
        if self.health_history['cpu_usage']:
            axes[0, 1].plot(self.health_history['cpu_usage'], 'g-', marker='s')
            axes[0, 1].axhline(y=50, color='r', linestyle='--', label='Target CPU')
            axes[0, 1].set_title('CPU Usage')
            axes[0, 1].set_xlabel('Monitoring Points')
            axes[0, 1].set_ylabel('CPU %')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Memory usage
        if self.health_history['memory_usage']:
            axes[1, 0].plot(self.health_history['memory_usage'], 'r-', marker='^')
            axes[1, 0].axhline(y=90, color='r', linestyle='--', label='Danger Zone')
            axes[1, 0].set_title('Memory Usage')
            axes[1, 0].set_xlabel('Monitoring Points')
            axes[1, 0].set_ylabel('Memory %')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate
        if self.health_history['learning_rates']:
            axes[1, 1].plot(self.health_history['learning_rates'], 'm-', marker='d')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Monitoring Points')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_health_monitor.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_continuous_monitoring(self, check_interval=300):  # 5 minutes
        """Run continuous monitoring"""
        logger.info(f"🔄 Starting continuous monitoring (check every {check_interval}s)")
        
        try:
            while True:
                metrics, alerts = self.generate_health_report()
                
                # Plot curves
                self.plot_training_curves()
                
                # Check for critical alerts
                critical_alerts = [alert for alert in alerts if "🚨" in alert]
                if critical_alerts:
                    logger.warning("🚨 CRITICAL ALERTS DETECTED!")
                    for alert in critical_alerts:
                        logger.warning(f"   {alert}")
                
                # Save health history
                with open('training_health_history.json', 'w') as f:
                    # Convert datetime objects to strings for JSON serialization
                    history_copy = self.health_history.copy()
                    history_copy['timestamps'] = [t.isoformat() for t in history_copy['timestamps']]
                    json.dump(history_copy, f, indent=2)
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")

def main():
    """Main monitoring function"""
    monitor = TrainingHealthMonitor()
    
    # Run single check
    logger.info("Running initial health check...")
    monitor.generate_health_report()
    
    # Ask user if they want continuous monitoring
    print("\n" + "="*60)
    print("🏥 TRAINING HEALTH MONITOR READY")
    print("="*60)
    print("Options:")
    print("1. Single health check (default)")
    print("2. Continuous monitoring (every 5 minutes)")
    print("3. Quick monitoring (every 1 minute)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "2":
        monitor.run_continuous_monitoring(300)  # 5 minutes
    elif choice == "3":
        monitor.run_continuous_monitoring(60)   # 1 minute
    else:
        logger.info("Single health check completed. Run again anytime!")

if __name__ == "__main__":
    main()