import time
import logging
from multiprocessing import Process
from utils.utils import get_result_folder, set_result_folder, create_logger
import subprocess


class GPUMonitor(Process):
    def __init__(self, logger_params, start_logging_time=0, log_interval=60):
        super(GPUMonitor, self).__init__(daemon=True)
        self.log_filepath = get_result_folder()

        self.logger = None
        self.logger_params = logger_params
        self.log_interval = log_interval
        self.start_logging_time = start_logging_time

    def run(self):

        time.sleep(self.start_logging_time)

        self._init_logger()

        while True:
            self._print_log()
            time.sleep(self.log_interval)

    def _init_logger(self):
        set_result_folder(self.log_filepath)
        create_logger(**self.logger_params)
        self.logger = logging.getLogger()

    def _print_log(self):
        gpu_logs = subprocess.check_output('nvidia-smi').decode('UTF-8').split('\n')
        for glog in gpu_logs:
            if glog == '':
                continue
            glog = glog.rstrip('\r').rstrip('\a')
            self.logger.info('{}'.format(glog))

