import numpy as np
import csv
from opendeep.utils.file_ops import find_files
import itertools
from opendeep.data import Dataset
from opendeep.models import RNN, SoftmaxLayer, Prototype
from opendeep.optimization import RMSProp
import theano.tensor as T
from opendeep import config_root_logger

def process_data(path):
    if 'bad' in path:
        label=0
    else:
        label=1
    for fpath in find_files(path):
        with open(fpath, 'r') as f:
            data = list(csv.reader(f))
        d = []
        for timestep in data:
            t = []
            for joint in timestep:
                angles = joint.split(';')
                angles = [float(angle) for angle in angles]
                t.append(angles)
            d.append(np.asarray(t, dtype='float32').flatten())
        yield (d, label)

def main():

    train_data = list(itertools.chain(process_data('data/bad/'), process_data('data/good/')))
    train_inputs = tuple(np.asarray(data) for data, label in train_data)
    train_targets = np.vstack([np.asarray(label, dtype='float32') for data, label in train_data])

    config_root_logger()

    dataset = Dataset(train_inputs, train_targets)

    rnn = RNN(input_size=12, hidden_size=6, layers=1)
    mean_pool = T.mean(rnn.get_hiddens(), axis=0)
    classification = SoftmaxLayer(inputs_hook=(6, mean_pool), output_size=1, out_as_probs=True)

    proto = Prototype(layers=[rnn, classification])

    opt = RMSProp(dataset, proto, batch_size=1)

    opt.train()

if __name__ == '__main__':
    main()
