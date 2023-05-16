import os
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchninja.core.utils.utils import to_device
from torchninja.core.utils.logger import Logger
from torchninja.core.utils.metrics import Metrics


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        data_loaders,
        model_args=None,
        optimizer_args=None,
        criterion=None,
        scheduler=None,
        use_mixed_precision=True,
        model_name=None,
        run_name=None,
        device=None,
        output_dir='output',
        save_checkpoints='last',
        checkpoints_dir='checkpoints',
        use_tqdm=True,
        log_to='both',
        log_dir='.',
        use_tensorboard=False,
        tensorboard_dir='runs',
        save_metrics=True,
        save_curves=True
    ):
        if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

        self.model_name = model_name or model.__class__.__name__
        self.run_name = run_name or self.model_name+'_'+datetime.now().strftime('%b-%d_%H-%M')

        self.model_args = model_args
        self.device = device

        # Check if model is a class or an instance
        if type(model) == type:
            self.model = model(**self.model_args).to(self.device)
            self.optimizer = optimizer(self.model.parameters(), **optimizer_args)
        else:
            self.model = model.to(self.device)
            self.optimizer = optimizer
        self.model.device = self.device
        
        # Training
        self.criterion = criterion
        self.scheduler = scheduler
        self.use_mixed_precision = use_mixed_precision
        self.data_loaders = data_loaders
        self.output_dir = output_dir
        self.save_metrics = save_metrics
        self.save_curves = save_curves

        # Checkpoints
        self.save_checkpoints = save_checkpoints
        self.checkpoints_dir = os.path.join(self.output_dir, checkpoints_dir)

        # Logging
        self.log_dir = os.path.join(self.output_dir, log_dir)
        self.logger = Logger(log_to=log_to, log_dir=self.log_dir, run_name=self.run_name)
        self.use_tqdm = use_tqdm

        # Tensorbaord
        self.use_tensorboard = use_tensorboard
        self.tensorboard_dir = os.path.join(self.output_dir, tensorboard_dir)
        if self.use_tensorboard:
            tb_logdir = os.path.join(self.tensorboard_dir, self.run_name)
            self.tb_writer = SummaryWriter(tb_logdir)

    # HOOKS

    def before_train(self, *args, **kwargs):
        pass

    def after_train(self, *args, **kwargs):
        pass

    def before_epoch(self, *args, **kwargs):
        pass

    def after_epoch(self, *args, **kwargs):
        pass

    def before_test(self, *args, **kwargs):
        pass

    def after_test(self, *args, **kwargs):
        pass

    # CHECKPOINTS

    def save_checkpoint(self, epoch):
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        if self.save_checkpoints.startswith('every:'):
            every = int(self.save_checkpoints.split(':')[1])
            if epoch % every == 0:
                filename = f'{self.run_name}_{epoch}.pt'
            else:
                filename = None
        elif self.save_checkpoints == 'all':
            filename = f'{self.run_name}_{epoch}.pt'
        elif self.save_checkpoints == 'last':
            filename = f'{self.run_name}.pt'
        else:
            filename = f'{self.run_name}.pt'
        
        if filename is not None:
            checkpoint_filepath = os.path.join(self.checkpoints_dir, filename)
            model_dict = dict()
            model_dict['epoch'] = epoch
            model_dict['model_state_dict'] = self.model.state_dict()
            model_dict['optimizer_state_dict'] = self.optimizer.state_dict()
            if self.model_args is not None:
                model_dict['model_args'] = self.model_args
            torch.save(model_dict, checkpoint_filepath)
            # self.logger.info(f'Checkpoint saved at {checkpoint_filepath}.')

    def load_checkpoint(self, run_name):
        if os.path.isfile(run_name):
            checkpoint_filepath = run_name
        else:
            checkpoint_filepath = os.path.join(self.checkpoints_dir,  f'{run_name}.pt')

        checkpoint = torch.load(checkpoint_filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f'Loaded {run_name} checkpoint at epoch {checkpoint["epoch"]}.')

    # TRAINING

    def make_pbar(self, iterable):
        return tqdm(iterable, leave=False, disable=not self.use_tqdm)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def train(self, epochs):
        self.before_train()
        self.logger.info(f'Starting run {self.run_name}: {self.model_name} with {epochs} epochs in {self.device}...')
        phases = list(self.data_loaders.keys() - {'test'})
        training_metrics = Metrics()

        scaler = torch.cuda.amp.GradScaler(enabled=self.use_mixed_precision)

        for epoch in range(epochs):
            self.before_epoch(epoch=epoch)

            for phase in phases:
                # Training phase
                if phase == 'train':
                    self.model.train()
                    pbar = self.make_pbar(self.data_loaders[phase])
                    for batch_idx, batch in enumerate(pbar):
                        batch = to_device(batch, self.device)
                        self.optimizer.zero_grad()

                        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=self.use_mixed_precision):
                            metrics = self.training_step(batch, batch_idx)
                            loss = metrics['loss']

                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                        pbar.set_description(f'[{epoch+1}/{epochs}] loss: {loss.item():.3f}')
                        training_metrics.add_batch_metrics(phase, metrics)
                # Validation phase
                else:
                    self.model.eval()
                    with torch.no_grad():
                        pbar = self.make_pbar(self.data_loaders[phase])
                        for batch_idx, batch in enumerate(pbar):
                            batch = to_device(batch, self.device)
                            metrics = self.validation_step(batch, batch_idx)
                            loss = metrics['loss']
                            pbar.set_description(f'[{epoch+1}/{epochs}] loss: {loss.item():.3f}')
                            training_metrics.add_batch_metrics(phase, metrics)

                epoch_metrics = training_metrics.aggretate_batches()

                # Apply schedular
                if self.scheduler is not None and phase == 'train':
                    if isinstance(self.scheduler, list) or isinstance(self.scheduler, tuple):
                        for scheduler in self.scheduler:
                            scheduler.step()
                    else:
                        self.scheduler.step()

                # Log metrics in Tensorboard
                if self.use_tensorboard:
                    for metric in epoch_metrics[phase]:
                        self.tb_writer.add_scalar(f'{phase}/{metric}', epoch_metrics[phase][metric], epoch)
                    
                    for name, weight in self.model.named_parameters():
                        self.tb_writer.add_histogram(name, weight, epoch)

            self.logger.epoch(epoch, epoch_metrics)

            if self.save_checkpoints is not None:
                self.save_checkpoint(epoch+1)
            
            self.after_epoch(epoch=epoch, metrics=epoch_metrics)

        self.logger.info('Finished training.')

        if self.use_tensorboard:
            self.tb_writer.close()
        
        # Save training metrics
        if self.save_metrics:
            training_metrics.save_json(os.path.join(self.output_dir, f'{self.run_name}.json'))

        # Save training curves
        if self.save_curves:
            training_metrics.save_plot(os.path.join(self.output_dir, f'{self.run_name}.png'))

        self.after_train(metrics=training_metrics)

        return training_metrics.values

    
    def test(self, test_data_loader=None, reduction='none'):
        self.before_test()
        self.logger.info('Starting test...')
        test_metrics = Metrics()

        if test_data_loader is None:
            test_data_loader = self.data_loaders['test']

        self.model.eval()
        with torch.no_grad():
            pbar = self.make_pbar(test_data_loader)
            for batch_idx, batch in enumerate(pbar):
                batch = to_device(batch, self.device)
                metrics = self.test_step(batch, batch_idx)
                test_metrics.add_batch_metrics('test', metrics)
            test_metrics.aggretate_batches(aggretation='concat')

        reduced_output = test_metrics.reduce(reduction)['test']

        log_metrics = reduced_output if reduction == 'mean' else test_metrics.reduce('mean')['test']
        self.logger.test(log_metrics)

        self.after_test()

        return reduced_output
