import json
import numpy as np
import torch
import matplotlib.pyplot as plt


class Metrics:
    def __init__(self):
        self.batch_values = {}
        self.values = {}

    def add_batch_metrics(self, phase, metrics):
        for metric, value in metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[metric] = value.detach().cpu()
                
                if value.dim() == 0:
                    metrics[metric] = metrics[metric].item()
                else:
                    metrics[metric] = metrics[metric].tolist()

        if phase not in self.batch_values:
            self.batch_values[phase] = {}
        
        for metric, value in metrics.items():
            if metric not in self.batch_values[phase]:
                self.batch_values[phase][metric] = []
            self.batch_values[phase][metric].append(value)

        return self.batch_values

    def aggretate_batches(self, aggretation='mean'):
        epoch_metrics = self.batch_values.copy()
        self.batch_values = {}
        for phase in epoch_metrics.keys():
            for metric, values in epoch_metrics[phase].items():
                if phase not in self.values:
                    self.values[phase] = {}
                if metric not in self.values[phase]:
                    self.values[phase][metric] = []

                # Flatten list of lists
                if isinstance(values[0], list):
                    values = [item for sublist in values for item in sublist]

                aggretation_funcs = {
                    'mean': np.mean,
                    'median': np.median,
                    'sum': np.sum,
                    'max': np.max,
                    'min': np.min,
                    'concat': lambda x: x
                }

                agg_func = aggretation_funcs[aggretation]
                epoch_metrics[phase][metric] = agg_func(values)

                if aggretation == 'concat':
                    self.values[phase][metric] += epoch_metrics[phase][metric]
                else:
                    self.values[phase][metric].append(epoch_metrics[phase][metric])

        return epoch_metrics
    
    def save_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.values, f, indent=4)

    def save_plot(self, path):
        phases = list(self.values.keys())
        metrics = list(self.values[phases[0]].keys())
        epochs = range(1, len(self.values[phases[0]][metrics[0]]) + 1)

        if len(metrics) == 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            metric = metrics[0]
            for phase in phases:
                ax.plot(epochs, self.values[phase][metric], label=f'{phase} {metric}')

            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_xlabel('epoch')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} over time')
            ax.grid('on', linestyle='--', linewidth=0.5, color='black', alpha=0.3)
            ax.legend()
        else:
            ncols = 2
            nrows = (len(metrics) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4*nrows))

            for i, metric in enumerate(metrics):
                row = i // ncols
                col = i % ncols
                ax = axes[row, col] if nrows > 1 else axes[col]

                for phase in phases:
                    ax.plot(epochs, self.values[phase][metric], label=f'{phase} {metric}')

                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                ax.set_xlabel('epoch')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} over time')
                ax.grid('on', linestyle='--', linewidth=0.5, color='black', alpha=0.3)
                ax.legend()

        plt.tight_layout()
        plt.savefig(path)

    def reduce(self, reduction='mean'):
        reduction_funcs = {
            'mean': np.mean,
            'median': np.median,
            'min': np.min,
            'max': np.max,
            'none': lambda x: x
        }

        if isinstance(reduction, str):
            reduction_func = reduction_funcs[reduction]
        elif isinstance(reduction, function):
            reduction_func = reduction
        else:
            reduction_func = reduction_funcs['none']

        reduced_output = {}
        for phase, metrics in self.values.items():
            reduced_output[phase] = {}
            for metric, values in metrics.items():
                reduced_output[phase][metric] = reduction_func(values)

        return reduced_output

    def __getitem__(self, keys):
        if isinstance(keys, str):
            keys = (keys,)
        values = self.values
        for key in keys:
            values = values[key]
        return values
    
    def __repr__(self):
        return str(self.values)
