import numpy as np
import csv
from opendeep.utils.file_ops import find_files
from opendeep.data import Dataset
from opendeep.models import LSTM, RNN
from opendeep.monitor import Monitor, Plot
from opendeep.optimization import RMSProp
import theano.tensor as T
from opendeep import config_root_logger

def process_data(path):
    for fpath in find_files(path):
        print(fpath)
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
        yield d

def main():
    # config_root_logger()

    train_inputs = []
    train_targets = []
    for swing in process_data('data/good/'):
        train_inputs.append(swing[:-1])
        train_targets.append(swing[1:])

    train_inputs = tuple(np.asarray(data, dtype='float32') for data in train_inputs)
    train_targets = tuple(np.asarray(data, dtype='float32') for data in train_targets)

    dataset = Dataset(train_inputs, train_targets)

    rnn = RNN(input_size=12, hidden_size=24, output_size=12, activation='tanh', noise_level=0.5)

    plot = Plot('test', open_browser=True)
    opt = RMSProp(dataset, rnn, batch_size=1, epochs=100, learning_rate=1e-4)

    opt.train(plot=plot)

    print("+++++++++++++++++++++ BAD SWINGS")
    mses = []
    import pprint
    positions = {'wrist': 0, 'elbow': 1, 'shoulder': 2}
    positions_inv = {v:k for k,v in positions.items()}
    pp = pprint.PrettyPrinter(indent=1)
    for swing in process_data('data/bad/'):
        ins = np.asarray([swing[:-1]])
        correct = np.asarray([swing[1:]])
        outs = rnn.run(ins)
        s = np.square(outs-correct)[0]
        idxs = []
        for timestep in s:

            wrist = np.mean(timestep[:3])
            elbow = np.mean(timestep[3:6])
            shoulder = np.mean(timestep[6:])
            bad = np.argmax([wrist, elbow, shoulder])
            idxs.append((positions_inv.get(bad), np.max([wrist, elbow, shoulder])))
        mse = np.square(outs-correct)
        pp.pprint(idxs)

    # print("+++++++++++++++++++++ GOOD SWINGS")
    # mses = []
    # for swing in process_data('data/good/'):
    #     ins = np.asarray([swing[:-1]])
    #     correct = np.asarray([swing[1:]])
    #     outs = rnn.run(ins)
    #     mse = np.mean(np.square(outs - correct))
    #     mses.append(mse)
    #     print(mse)
    # print("avg:", np.mean(mses))

if __name__ == '__main__':
    main()
