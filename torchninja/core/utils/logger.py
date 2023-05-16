import os
import sys
import logging


class Logger:
    def __init__(self, log_to='both', log_dir='.', run_name='run'):
        self.log_to = log_to
        self.log_dir = log_dir
        self.run_name = run_name
        
        self.logger = logging.getLogger('torchninja')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.propagate = False

        if self.log_to == 'both' or self.log_to == 'file':
            self.set_file_logger()
        if self.log_to == 'both' or self.log_to == 'console':
            self.set_stream_logger()

    def set_stream_logger(self):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(sh)

    def set_file_logger(self):
        os.makedirs(self.log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(self.log_dir, self.run_name + '.log'))
        fh.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
        self.logger.addHandler(fh)

    def epoch(self, epoch, metrics):
        values = []
        for phase in metrics.keys():
            for metric, value in metrics[phase].items():
                values.append(f'{phase}_{metric}: {value:.5f}')
        values = ', '.join(values)
        self.logger.info(f'Epoch {epoch}: {values}')

    def test(self, metrics):
        log_text = [f'{metric}: {value:.5f}' for metric, value in metrics.items()]
        log_text = ', '.join(log_text)
        self.logger.info(f'Test results (avg): {log_text}')

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def exception(self, message):
        self.logger.exception(message)
