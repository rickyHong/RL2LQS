##########################################################################################
# Human Resource Management Learning Framework
# Adapted from RL2LQS for HR optimization and prediction
##########################################################################################

import os
import sys
from datetime import datetime
import pytz
import logging

# Machine Environment Config
DEBUG_MODE = False
ENABLE_GPU_MONITOR = False
USE_CUDA = True

if USE_CUDA:
    CUDA_DEVICE_NUM = [0, 1]  # GPU numbers
else:
    CUDA_DEVICE_NUM = [1, 2, 3, 4, 5, 6, 7, 8]  # CPU numbers

##########################################################################################
# Path Configuration
os.path.dirname(os.path.abspath("__file__"))
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

##########################################################################################
# Logger Parameters
process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))
result_folder_template = './result/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'

logger_params = {
    'log_file': {
        'desc': 'hrm_learning',
        'filename': 'hrm_log.txt',
        'filepath': result_folder_template,
    }
}

##########################################################################################
# HRM Training Parameters

# Environment parameters
env_params = {
    'num_employees': 1000,
    'num_departments': 10,
    'skill_dimensions': 20,
    'time_horizon': 12,  # months
}

# Training parameters
batch_size = 64
training_params = {
    'learning_mode': 'actor_critic',  # 'actor_critic', 'ppo', 'evolution'
    'buffer_size': 10000,
    'clip_epsilon': 0.2,  # For PPO
    'ppo_epochs': 4,      # For PPO
    'population_size': 20,  # For evolution strategy
}

# Model parameters
model_params = {
    'load': False,
    'model_path': '',
    'hidden_dim1': 128,
    'hidden_dim2': 256,
    'hidden_dim3': 128,
    'final_dim': 64,
    'critic_dim1': 128,
    'critic_dim2': 256,
    'critic_dim3': 128,
    'critic_final_dim': 64,
    'num_revenue_classes': 3,
    'num_satisfaction_classes': 5,
    'num_objectives': 5,
    'parameter_dim': 10,
    'hr_strategies': {
        'recruitment': 5,
        'training': 8,
        'compensation': 6,
        'culture': 6,
        'performance': 5,
        'leadership': 5,
        'technology': 4,
        'restructuring': 4
    }
}

# Optimizer parameters
optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
}

# Run parameters
run_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'batch_size': batch_size,
    'episodes': batch_size * 1000,  # Total training episodes
    'evaluation_episodes': 100,
    'save_interval': 500,
    'logging': {
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_none.json'
        },
    },
}

##########################################################################################
# Different Training Modes

def run_actor_critic_training():
    """Run Actor-Critic training mode"""
    training_params['learning_mode'] = 'actor_critic'
    return train_hrm_model()

def run_ppo_training():
    """Run PPO training mode"""
    training_params['learning_mode'] = 'ppo'
    return train_hrm_model()

def run_evolution_training():
    """Run Evolution Strategy training mode"""
    training_params['learning_mode'] = 'evolution'
    return train_hrm_model()

def run_multi_objective_training():
    """Run multi-objective optimization training"""
    # Modify parameters for multi-objective learning
    model_params['num_objectives'] = 5
    training_params['learning_mode'] = 'actor_critic'
    return train_hrm_model()

def run_meta_learning():
    """Run meta-learning across multiple organizations"""
    # Increase diversity in environment
    env_params['num_employees'] = [500, 1000, 1500, 2000]  # Different org sizes
    env_params['market_variability'] = True
    return train_hrm_model()

##########################################################################################
# Main Training Function

def train_hrm_model():
    """Main HRM model training function"""
    from utils.utils import create_logger, copy_all_src
    from utils.gpumonitor import GPUMonitor
    from HRM_Trainer import HRMTrainer
    
    # Set debug mode if enabled
    if DEBUG_MODE:
        run_params['episodes'] = batch_size * 10
    
    # Create logger
    create_logger(**logger_params)
    logger = logging.getLogger('root')
    
    # Print configuration
    logger.info('=== HRM Learning Framework Started ===')
    logger.info(f'DEBUG_MODE: {DEBUG_MODE}')
    logger.info(f'USE_CUDA: {USE_CUDA}, CUDA_DEVICE_NUM: {CUDA_DEVICE_NUM}')
    logger.info(f'Learning Mode: {training_params["learning_mode"]}')
    logger.info(f'Episodes: {run_params["episodes"]}')
    logger.info(f'Batch Size: {batch_size}')
    
    # Log all parameters
    for param_name in ['env_params', 'training_params', 'model_params', 'optimizer_params', 'run_params']:
        logger.info(f'{param_name}: {globals()[param_name]}')
    
    # Initialize trainer
    trainer = HRMTrainer(
        env_params=env_params,
        training_params=training_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        run_params=run_params
    )
    
    # Copy source code to results folder
    copy_all_src(trainer.result_folder)
    
    # Start GPU monitoring if enabled
    if ENABLE_GPU_MONITOR:
        gpu_logger_params = logger_params.copy()
        gpu_logger_params['log_file']['filename'] = 'gpu_monitor.txt'
        gm = GPUMonitor(gpu_logger_params, start_logging_time=20, log_interval=600)
        gm.start()
    
    # Run training
    logger.info('Starting HRM training...')
    trainer.run()
    
    # Evaluate trained model
    logger.info('Evaluating trained model...')
    evaluation_results = trainer.evaluate(num_episodes=run_params['evaluation_episodes'])
    
    # Log evaluation results
    logger.info('=== Evaluation Results ===')
    for metric, value in evaluation_results.items():
        logger.info(f'{metric}: {value}')
    
    # Generate final report
    generate_final_report(trainer, evaluation_results)
    
    logger.info('=== HRM Learning Framework Completed ===')
    
    return trainer, evaluation_results

##########################################################################################
# Specialized Training Functions

def train_revenue_prediction():
    """Train model specifically for revenue prediction"""
    logger = logging.getLogger('root')
    logger.info('=== Revenue Prediction Training ===')
    
    # Modify parameters for revenue focus
    model_params['num_revenue_classes'] = 5  # More granular revenue classes
    training_params['revenue_weight'] = 0.7
    training_params['satisfaction_weight'] = 0.3
    
    return train_hrm_model()

def train_satisfaction_prediction():
    """Train model specifically for customer satisfaction prediction"""
    logger = logging.getLogger('root')
    logger.info('=== Customer Satisfaction Prediction Training ===')
    
    # Modify parameters for satisfaction focus
    model_params['num_satisfaction_classes'] = 7  # More granular satisfaction scale
    training_params['revenue_weight'] = 0.3
    training_params['satisfaction_weight'] = 0.7
    
    return train_hrm_model()

def train_comprehensive_model():
    """Train comprehensive model for all HR objectives"""
    logger = logging.getLogger('root')
    logger.info('=== Comprehensive HR Model Training ===')
    
    # Balanced parameters for all objectives
    training_params['learning_mode'] = 'ppo'  # PPO for stable learning
    model_params['num_objectives'] = 8  # More objectives
    run_params['episodes'] = batch_size * 2000  # Longer training
    
    return train_hrm_model()

##########################################################################################
# Hyperparameter Optimization

def hyperparameter_search():
    """Perform hyperparameter optimization"""
    import itertools
    
    logger = logging.getLogger('root')
    logger.info('=== Hyperparameter Search ===')
    
    # Define search space
    learning_rates = [1e-5, 1e-4, 1e-3]
    batch_sizes = [32, 64, 128]
    hidden_dims = [64, 128, 256]
    
    best_performance = float('-inf')
    best_params = None
    
    # Grid search
    for lr, batch_sz, hidden_dim in itertools.product(learning_rates, batch_sizes, hidden_dims):
        logger.info(f'Testing: lr={lr}, batch_size={batch_sz}, hidden_dim={hidden_dim}')
        
        # Update parameters
        optimizer_params['optimizer']['lr'] = lr
        run_params['batch_size'] = batch_sz
        model_params['hidden_dim1'] = hidden_dim
        model_params['hidden_dim2'] = hidden_dim * 2
        
        # Reduce episodes for faster search
        original_episodes = run_params['episodes']
        run_params['episodes'] = batch_sz * 200
        
        try:
            trainer, results = train_hrm_model()
            performance = results['average_reward']
            
            if performance > best_performance:
                best_performance = performance
                best_params = {
                    'lr': lr,
                    'batch_size': batch_sz,
                    'hidden_dim': hidden_dim,
                    'performance': performance
                }
                
            logger.info(f'Performance: {performance:.4f}')
            
        except Exception as e:
            logger.error(f'Error in hyperparameter combination: {e}')
        
        # Restore original episodes
        run_params['episodes'] = original_episodes
    
    logger.info(f'Best hyperparameters: {best_params}')
    return best_params

##########################################################################################
# Results Analysis and Reporting

def generate_final_report(trainer, evaluation_results):
    """Generate comprehensive final report"""
    import json
    import pandas as pd
    
    logger = logging.getLogger('root')
    result_folder = trainer.result_folder
    
    # Create comprehensive report
    report = {
        'training_summary': {
            'total_episodes': len(trainer.training_history['episode_rewards']),
            'best_reward': trainer.best_reward,
            'final_reward': trainer.training_history['episode_rewards'][-1] if trainer.training_history['episode_rewards'] else 0,
            'training_mode': training_params['learning_mode']
        },
        'evaluation_results': evaluation_results,
        'configuration': {
            'env_params': env_params,
            'training_params': training_params,
            'model_params': model_params,
            'optimizer_params': optimizer_params
        },
        'model_architecture': {
            'actor_parameters': sum(p.numel() for p in trainer.actor.parameters()),
            'critic_parameters': sum(p.numel() for p in trainer.critic.parameters()),
            'predictor_parameters': sum(p.numel() for p in trainer.predictor.parameters())
        }
    }
    
    # Save main report
    with open(f"{result_folder}/final_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate performance summary
    if trainer.training_history['hr_metrics']:
        performance_summary = analyze_hr_performance(trainer.training_history['hr_metrics'])
        
        with open(f"{result_folder}/performance_summary.json", 'w') as f:
            json.dump(performance_summary, f, indent=2)
    
    logger.info(f'Final report saved to {result_folder}')

def analyze_hr_performance(hr_metrics_history):
    """Analyze HR performance over time"""
    # Calculate trends and improvements
    initial_metrics = hr_metrics_history[0]
    final_metrics = hr_metrics_history[-1]
    
    improvements = {
        'satisfaction_improvement': final_metrics.employee_satisfaction - initial_metrics.employee_satisfaction,
        'productivity_improvement': final_metrics.productivity_index - initial_metrics.productivity_index,
        'retention_improvement': final_metrics.retention_rate - initial_metrics.retention_rate,
        'innovation_improvement': final_metrics.innovation_index - initial_metrics.innovation_index,
        'culture_improvement': final_metrics.organizational_culture_score - initial_metrics.organizational_culture_score
    }
    
    # Calculate target achievement rates
    targets = {
        'employee_satisfaction': 0.85,
        'productivity_index': 1.2,
        'retention_rate': 0.90,
        'innovation_index': 0.75,
        'organizational_culture_score': 0.85
    }
    
    achievement_rates = {
        'satisfaction_achievement': (final_metrics.employee_satisfaction / targets['employee_satisfaction']) * 100,
        'productivity_achievement': (final_metrics.productivity_index / targets['productivity_index']) * 100,
        'retention_achievement': (final_metrics.retention_rate / targets['retention_rate']) * 100,
        'innovation_achievement': (final_metrics.innovation_index / targets['innovation_index']) * 100,
        'culture_achievement': (final_metrics.organizational_culture_score / targets['organizational_culture_score']) * 100
    }
    
    return {
        'improvements': improvements,
        'achievement_rates': achievement_rates,
        'initial_metrics': {
            'satisfaction': initial_metrics.employee_satisfaction,
            'productivity': initial_metrics.productivity_index,
            'retention': initial_metrics.retention_rate,
            'innovation': initial_metrics.innovation_index,
            'culture': initial_metrics.organizational_culture_score
        },
        'final_metrics': {
            'satisfaction': final_metrics.employee_satisfaction,
            'productivity': final_metrics.productivity_index,
            'retention': final_metrics.retention_rate,
            'innovation': final_metrics.innovation_index,
            'culture': final_metrics.organizational_culture_score
        }
    }

##########################################################################################
# Demonstration and Testing Functions

def run_demo():
    """Run a quick demonstration of the HRM framework"""
    global run_params, training_params
    
    logger = logging.getLogger('root')
    logger.info('=== HRM Framework Demo ===')
    
    # Reduce parameters for quick demo
    original_episodes = run_params['episodes']
    run_params['episodes'] = batch_size * 50
    env_params['num_employees'] = 200
    training_params['learning_mode'] = 'actor_critic'
    
    try:
        trainer, results = train_hrm_model()
        
        logger.info('Demo completed successfully!')
        logger.info(f'Demo results: {results}')
        
        # Test prediction capabilities
        test_predictions(trainer)
        
    finally:
        # Restore original parameters
        run_params['episodes'] = original_episodes
        env_params['num_employees'] = 1000

def test_predictions(trainer):
    """Test the prediction capabilities of the trained model"""
    import torch
    
    logger = logging.getLogger('root')
    logger.info('=== Testing Prediction Capabilities ===')
    
    # Generate test scenarios
    test_scenarios = [
        "High satisfaction, high productivity organization",
        "Low satisfaction, low productivity organization", 
        "Medium satisfaction, high innovation organization"
    ]
    
    trainer.predictor.eval()
    
    for i, scenario in enumerate(test_scenarios):
        # Create test state (simplified)
        test_state = trainer.env.get_state()
        test_state_tensor = torch.FloatTensor(test_state).unsqueeze(0).to(trainer.device)
        
        with torch.no_grad():
            predictions = trainer.predictor(test_state_tensor)
            
            revenue_pred = torch.argmax(predictions['revenue_probs']).item()
            satisfaction_pred = torch.argmax(predictions['satisfaction_probs']).item()
            
            logger.info(f'Scenario {i+1}: {scenario}')
            logger.info(f'  Predicted Revenue Class: {revenue_pred}')
            logger.info(f'  Predicted Satisfaction Class: {satisfaction_pred}')

##########################################################################################
# Main Execution Functions

def main():
    """Main execution function"""
    if DEBUG_MODE:
        run_demo()
    else:
        # Default comprehensive training
        train_comprehensive_model()

def main_actor_critic():
    """Main function for Actor-Critic training"""
    run_actor_critic_training()

def main_ppo():
    """Main function for PPO training"""
    run_ppo_training()

def main_evolution():
    """Main function for Evolution Strategy training"""
    run_evolution_training()

def main_revenue_prediction():
    """Main function for revenue prediction training"""
    train_revenue_prediction()

def main_satisfaction_prediction():
    """Main function for satisfaction prediction training"""
    train_satisfaction_prediction()

def main_hyperparameter_search():
    """Main function for hyperparameter optimization"""
    from utils.utils import create_logger
    create_logger(**logger_params)
    hyperparameter_search()

def main_demo():
    """Main function for demo"""
    from utils.utils import create_logger
    create_logger(**logger_params)
    run_demo()

##########################################################################################
# Entry Point

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HRM Learning Framework')
    parser.add_argument('--mode', type=str, default='comprehensive',
                       choices=['comprehensive', 'actor_critic', 'ppo', 'evolution', 
                               'revenue', 'satisfaction', 'hyperparameter', 'demo'],
                       help='Training mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU monitoring')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    
    args = parser.parse_args()
    
    # Update global parameters based on arguments
    if args.debug:
        DEBUG_MODE = True
    if args.gpu:
        ENABLE_GPU_MONITOR = True
    if args.episodes:
        run_params['episodes'] = args.episodes
    if args.batch_size:
        run_params['batch_size'] = args.batch_size
        batch_size = args.batch_size
    
    # Select execution mode
    if args.mode == 'comprehensive':
        main()
    elif args.mode == 'actor_critic':
        main_actor_critic()
    elif args.mode == 'ppo':
        main_ppo()
    elif args.mode == 'evolution':
        main_evolution()
    elif args.mode == 'revenue':
        main_revenue_prediction()
    elif args.mode == 'satisfaction':
        main_satisfaction_prediction()
    elif args.mode == 'hyperparameter':
        main_hyperparameter_search()
    elif args.mode == 'demo':
        main_demo()
    else:
        print(f"Unknown mode: {args.mode}")
        parser.print_help()