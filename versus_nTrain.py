# versus nTrain
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import TensorDataset, DataLoader
from numpy.fft import fft
from sklearn.manifold import TSNE
from scipy.signal import stft, windows
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import random
import matplotlib as mpl
from scipy.interpolate import make_interp_spline
import torch.nn.functional as F

# --- Device setting ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_random_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(0)  

# --- Functions ---
def make_hankel_onlyS(observation):
    l = len(observation)
    r = int(np.ceil(l / 2))
    c = l - r + 1
    m = np.zeros((r, c), dtype=np.complex128)
    for kk in range(r):
        m[kk, :] = observation[kk:kk + c]
    _, S, _ = np.linalg.svd(m, full_matrices=False)
    return S.real

def make_spectrogram(signal, nperseg=40, noverlap=36, sigma=1):
    real_smooth = gaussian_filter(signal.real, sigma=sigma)
    imag_smooth = gaussian_filter(signal.imag, sigma=sigma)
    filtered_signal = real_smooth + 1j * imag_smooth

    window = windows.hann(nperseg)
    _, _, Zxx = stft(filtered_signal, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nperseg, return_onesided=False)

    power = np.abs(Zxx) ** 2
    power_db = 10 * np.log10(power + 1e-12)

    power_db -= power_db.min()
    power_db /= (power_db.max() + 1e-8)
    power_db *= 255.0

    spec = resize(power_db, (100, 100), order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
    spec = np.stack([spec, spec, spec], axis=0)
    return spec

def get_real_imag_input(iq_data):
    x_real = np.real(iq_data).astype(np.float32)
    x_imag = np.imag(iq_data).astype(np.float32)
    
    #norm = np.sqrt(x_real**2 + x_imag**2 + 1e-9)
    #x_real /= norm
    #x_imag /= norm

    x = np.stack((x_real, x_imag), axis=1)  # shape: (N, 2, 128)
    return x



def get_amp_phase_input_norm(iq_data):
    amp = np.abs(iq_data).astype(np.float32)
    phase = np.angle(iq_data).astype(np.float32)
    x = np.stack((amp, phase), axis=1)  # shape: (N, 2, 128)
    return x


# --- Models ---
# fft proposed
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            #nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)

class SimpleMLP2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            #nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)

class Legacy1DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Legacy1DCNN, self).__init__()
        self.noise = NoiseLayer(snr_db=5) 
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),  # ➜ size: 64 × 64

            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),  # ➜ size: 32 × 32

            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Dropout(0.3),  # 16 x 32 = 512
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)



class TableICNN(nn.Module):
    def __init__(self, num_classes=4):
        super(TableICNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 12, 3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(12, 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
patience=20

# --- Training functions ---
# proposed abs
def train_eval_mlp(x_train, x_val, x_test, y_train, y_val, y_true, input_dim, hidden_dim, output_dim, epochs=120):
    set_random_seed(0)

    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    y_label_tensor = torch.argmax(y_train_tensor, dim=1)
    y_val_label_tensor = torch.argmax(y_val_tensor, dim=1)

    model = SimpleMLP2(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-5, amsgrad=False)
    criterion = nn.CrossEntropyLoss()
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, step_size=100, gamma=0.1)

    #g = torch.Generator()
    #g.manual_seed(0)

    train_loader = DataLoader(TensorDataset(x_train, y_label_tensor), batch_size=64, shuffle=True)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0


    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        #scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = criterion(val_pred, y_val_tensor).item()
            val_loss = criterion(val_pred, y_val_label_tensor).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_model)

    model.eval()
    with torch.no_grad():
        pred = model(x_test).argmax(dim=1).cpu().numpy()
    correct = (pred == y_true).sum()
    return correct / len(y_true)

def train_eval_mlp2(x_train, x_val, x_test, y_train, y_val, y_true, input_dim, hidden_dim, output_dim, epochs=120):
    set_random_seed(0)

    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    y_label_tensor = torch.argmax(y_train_tensor, dim=1)
    y_val_label_tensor = torch.argmax(y_val_tensor, dim=1)

    model = SimpleMLP2(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    #g = torch.Generator()
    #g.manual_seed(0)

    train_loader = DataLoader(TensorDataset(x_train, y_label_tensor), batch_size=64, shuffle=True)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model = None
    patience_counter = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        #scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = criterion(val_pred, y_val_tensor).item()
            val_loss = criterion(val_pred, y_val_label_tensor).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_model)


    model.eval()
    with torch.no_grad():
        pred = model(x_test).argmax(dim=1).cpu().numpy()
    correct = (pred == y_true).sum()
    return correct / len(y_true)

#SCNN2
def train_eval_cnn(x_train, x_val, x_test, y_train, y_val, y_true, output_dim, epochs=50):
    set_random_seed(0)
    
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

    model = TableICNN(output_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.0005, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.18)

    #g = torch.Generator()
    #g.manual_seed(0)

    train_loader = DataLoader(TensorDataset(x_train, y_train_tensor), batch_size=64, shuffle=True)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = criterion(val_pred, y_val_tensor).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_model)

    # Final Test
    model.eval()    
    with torch.no_grad():
        pred = model(x_test).argmax(dim=1).cpu().numpy()
    correct = (pred == y_true).sum()
    return correct / len(y_true)


# cnn legacy
class CNN2(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN2, self).__init__()
        #self.inst_norm = nn.InstanceNorm1d(num_features=2, affine=False)
        self.conv = nn.Sequential(
            #self.inst_norm,
            nn.Conv1d(2, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.LayerNorm([32, 128]),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.LayerNorm([32, 128]),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.MaxPool1d(2),
            nn.LayerNorm([32, 64]),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32 * 64, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


#CNN legacy
def train_eval_cnn2(x_train, x_val, x_test, y_train, y_val, y_test, num_classes, epochs=40):
    set_random_seed(0)

    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    model = CNN2(num_classes).to(device)
    #optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    #g = torch.Generator()
    #g.manual_seed(0)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)

    best_val_loss = float('inf')
    best_model = None
    patience = 20
    patience_counter = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = criterion(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        pred = model(x_test).argmax(dim=1).cpu().numpy()
    correct = (pred == y_test).sum()
    return correct / len(y_test)
   
# --- Main Experiment ---
#SNR_vec = np.arange(-20, 19, 4)
snr=10
#nTotal_per_mod = 1000
nTotal_per_mod_vec=[200,400,600,800,1000]
str_dim = 24 #proposed
modulations = ['BPSK', 'QAM16', 'QAM64', 'QPSK']
Qmax = len(modulations)

with open('RML2016.10a_dict.pkl', 'rb') as f:
    data_dict = pickle.load(f, encoding='latin1')

repeats = 3
results_all = {key: [] for key in ['Legacy_abs', 'Legacy_realimag', 'Proposed_hankel_abs', 'Proposed_hankel_fft_abs', 'CNN_spectrogram']}

for repeat_idx in range(repeats):
    print(f"Repeat {repeat_idx+1}/{repeats}")
    set_random_seed(repeat_idx)  

    results = {key: [] for key in results_all.keys()}

    for nTotal_per_mod in nTotal_per_mod_vec:
        X_train_all, X_val_all, X_test_all = [], [], []
        Y_train_all, Y_val_all, Y_test_all = [], [], []

        for idx, mod in enumerate(modulations):
            x = data_dict[(mod, snr)]
            iq = x[:, 0, :] + 1j * x[:, 1, :]
            iq = iq[:nTotal_per_mod]

            #np.random.seed(0)
            perm = np.random.permutation(len(iq))
            nTrain = int(0.7 * len(iq))
            nVal = int(0.1 * len(iq))

            idx_train = perm[:nTrain]
            idx_val = perm[nTrain:nTrain+nVal]
            idx_test = perm[nTrain+nVal:]

            X_train_all.append(iq[idx_train])
            X_val_all.append(iq[idx_val])
            X_test_all.append(iq[idx_test])

            Y_train_all.append(np.full(len(idx_train), idx))
            Y_val_all.append(np.full(len(idx_val), idx))
            Y_test_all.append(np.full(len(idx_test), idx))

        X_train = np.concatenate(X_train_all)
        X_val = np.concatenate(X_val_all)
        X_test = np.concatenate(X_test_all)
        Y_train = np.concatenate(Y_train_all)
        Y_val = np.concatenate(Y_val_all)
        Y_test = np.concatenate(Y_test_all)

        Y_train_onehot = np.eye(Qmax)[Y_train]
        Y_val_onehot = np.eye(Qmax)[Y_val]

        results['Legacy_abs'].append(train_eval_mlp(np.abs(X_train), np.abs(X_val), np.abs(X_test),
                                                    Y_train_onehot, Y_val_onehot, Y_test, 128, str_dim, Qmax))
        
        X_train1 = get_real_imag_input(np.concatenate(X_train_all))
        X_val1 = get_real_imag_input(np.concatenate(X_val_all))
        X_test1 = get_real_imag_input(np.concatenate(X_test_all))

        Y_train1 = np.concatenate(Y_train_all)
        Y_val1 = np.concatenate(Y_val_all)
        Y_test1 = np.concatenate(Y_test_all)

        results['Legacy_realimag'].append(train_eval_cnn2(X_train1, X_val1, X_test1, Y_train1, Y_val1, Y_test1, num_classes=Qmax))

        results['Proposed_hankel_abs'].append(train_eval_mlp(np.array([make_hankel_onlyS(x) for x in X_train]),
                                                             np.array([make_hankel_onlyS(x) for x in X_val]),
                                                             np.array([make_hankel_onlyS(x) for x in X_test]),
                                                             Y_train_onehot, Y_val_onehot, Y_test, 64, str_dim, Qmax))
        results['Proposed_hankel_fft_abs'].append(train_eval_mlp2(np.array([make_hankel_onlyS(fft(x)) for x in X_train]),
                                                                 np.array([make_hankel_onlyS(fft(x)) for x in X_val]),
                                                                 np.array([make_hankel_onlyS(fft(x)) for x in X_test]),
                                                                 Y_train_onehot, Y_val_onehot, Y_test, 64, str_dim, Qmax))
        results['CNN_spectrogram'].append(train_eval_cnn(np.array([make_spectrogram(x) for x in X_train]),
                                                         np.array([make_spectrogram(x) for x in X_val]),
                                                         np.array([make_spectrogram(x) for x in X_test]),
                                                         Y_train, Y_val, Y_test, Qmax))
    
  
    for key in results:
        results_all[key].append(results[key])


results_avg = {key: np.mean(results_all[key], axis=0) for key in results_all}

mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'  
mpl.rcParams['font.family'] = 'serif'    
keys = ['Proposed_hankel_abs', 'Proposed_hankel_fft_abs', 'Legacy_abs', 'Legacy_realimag', 'CNN_spectrogram']
labels = [
    'Proposed method (with original signal)',
    'Proposed method (with FFT-processed signal)',
    'Deep learning (MLP with absolute values of signal)',
    'Deep learning (CNN with real and imaginary values of signal)',
    'SCNN2'
]
markers = ['^', 'x', 'o', 's', 'h']  
train_counts = [int(N * 0.7 * 4) for N in nTotal_per_mod_vec]

plt.figure()

plt.plot(train_counts, results_avg['Proposed_hankel_abs'], label='Proposed method (with original signal)', marker='^')
plt.plot(train_counts, results_avg['Proposed_hankel_fft_abs'], label='Proposed method (with FFT-processed signal)', marker='x')
plt.plot(train_counts, results_avg['Legacy_abs'], label='Deep learning (MLP with absolute values of signal)', marker='o')
plt.plot(train_counts, results_avg['Legacy_realimag'], label='Deep learning (CNN with real and imaginary values of signal)', marker='s')
plt.plot(train_counts, results_avg['CNN_spectrogram'], label='SCNN2', marker='h')

plt.grid(True)
plt.xlabel('Number of training dataset')
plt.ylabel('Detection rate')
plt.xticks(train_counts)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlim([train_counts[0], train_counts[-1]])
plt.ylim([0, 1])
plt.legend(prop={'size': 8}, loc='best')
plt.show()
