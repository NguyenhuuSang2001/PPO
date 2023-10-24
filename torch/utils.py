import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def plot_result(data, show_result=False, title='Result', xlabel='Episode', ylabel='Reward'):
    plt.figure(1)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            
    
    data_t = [np.mean(data[max(0, i-100):(i+1)]) for i in range(len(data))]
    data_t = torch.tensor(data_t, dtype=torch.float)
    if show_result:
        plt.title(title)
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(data_t.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
