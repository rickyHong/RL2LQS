##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
ENABLE_GPU_MONITOR = False
USE_CUDA = True
if USE_CUDA:
    CUDA_DEVICE_NUM = [0,1] # gpu number
else:
    CUDA_DEVICE_NUM = [1,2,3,4,5,6,7,8,9,10] # cpu number

##########################################################################################
# Path Config
import os
import sys

#os.chdir(os.path.dirname(os.path.abspath(__file__))) # for .py
os.path.dirname(os.path.abspath("__file__")) # for .ipynb
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# Logger Parameters

from datetime import datetime
import pytz
process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))
result_folder_template = './result/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'

logger_params = {
    'log_file': {
        #'desc': 'kwon-refactoring_v0',
        'desc': 'jae',
        'filename': 'log.txt',
        'filepath': result_folder_template,
    }
}

##########################################################################################
# parameters

batch_size = 1000
run_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'PATH': 'trained_model.pt',
    'batch_size': batch_size,
    'episodes': batch_size*5000,
    'memory_window': 1,
    'logging': {
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_none.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_none.json'
        },
    },
}

es_params = {
    'ESmode': 'train', #'train' / 'inference' / 'inference_max' / 'static'
    'static_p1': -2.0,
    'static_p2': -1.0,
    'num_test_paras': 1,    # fix to 1
    'history_size': 1,      # fix to 1
    'sample_size': 5,       # the number of parameter samples k
    'target_success': 1e4,  # target success count C_T
    'halting': 3000         # maximum time step t_max
}

env_params = {
    'qubit_size': 1,
}

path  =""
filename = ""
model_params = {
    'load': False,
    'model_path':  path + filename,
    'reward': 'trajectory', #'count', #
    'history_size': es_params['history_size'],
    'para_space1': [0.0,-1.0,-2.0,-3.0],   # Action space for sampling range A_sigma in log scale
    'para_space2': [0.0,-1.0,-2.0,-3.0],   # Action space for learning rate A_eta in log scale
    'random_selection': 1000, #random_sample < batch_size
    'update': 500    # random_selection * update determines the size of mini-buffer
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
}

##########################################################################################
# main
import logging
from utils.utils import create_logger, copy_all_src
from utils.gpumonitor import GPUMonitor
from QTrainer import QTrainer as Trainer
#mport torch


def _set_debug_mode():
    global run_params
    #run_params['episodes'] = batch_size * 2

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()
    
    
    ###################
    ### PPO method ####
    ###################
    trainer = Trainer(env_params=env_params,
                      es_params=es_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      run_params=run_params)

    copy_all_src(trainer.result_folder)

    if ENABLE_GPU_MONITOR:
        GPU_Monitor_logger_params = logger_params
        GPU_Monitor_logger_params['log_file']['filename'] = 'log_gpu_monitor.txt'
        gm = GPUMonitor(GPU_Monitor_logger_params, start_logging_time=20, log_interval=600)
        gm.start()

    trainer.run()

##########################################################################################
if __name__ == "__main__":
    main()
