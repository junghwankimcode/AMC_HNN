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
   # set_random_seed(0)

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

   # g = torch.Generator()
   # g.manual_seed(0)

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
    return correct / len(y_true), model

def train_eval_mlp2(x_train, x_val, x_test, y_train, y_val, y_true, input_dim, hidden_dim, output_dim, epochs=120):
    #set_random_seed(0)

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
    return correct / len(y_true), model

#SCNN2
def train_eval_cnn(x_train, x_val, x_test, y_train, y_val, y_true, output_dim, epochs=50):
    #set_random_seed(0)

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
    return correct / len(y_true), model


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
    #set_random_seed(0)

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
    return correct / len(y_test), model

# --- Main Experiment ---
# --- Extract Feature for t-SNE Visualization ---
def extract_features(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        if isinstance(model, (CNN2, Legacy1DCNN)):
            features = model.conv(X_tensor) if hasattr(model, 'conv') else model.features(X_tensor)
            features = features.view(features.size(0), -1).cpu().numpy()
        elif isinstance(model, TableICNN):
            features = model.features(X_tensor).view(X_tensor.size(0), -1).cpu().numpy()
        else:  # MLP
            for name, layer in model.model.named_children():
                X_tensor = layer(X_tensor)
                if name == '4':  # just before final linear layer
                    break
            features = X_tensor.cpu().numpy()
    return features

# --- Visualize ---
'''
def visualize_tsne(features_list, labels, modulations, titles):
    colors = ['red', 'blue', 'green', 'orange']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for i, (features, title) in enumerate(zip(features_list, titles)):
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
        reduced = tsne.fit_transform(features)
        ax = axes[i]
        for j, mod in enumerate(modulations):
            ax.scatter(reduced[labels == j, 0], reduced[labels == j, 1], label=mod, alpha=0.6, s=10, color=colors[j])
        ax.set_title(title)
        ax.set_xlabel('t-SNE-1')
        ax.set_ylabel('t-SNE-2')
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.show()
'''
def visualize_tsne(features_list, labels, modulations, titles):
    colors = ['red', 'blue', 'green', 'orange']

    for i, (features, title) in enumerate(zip(features_list, titles)):
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
        reduced = tsne.fit_transform(features)

        plt.figure(figsize=(6, 5))
        for j, mod in enumerate(modulations):
            plt.scatter(reduced[labels == j, 0], reduced[labels == j, 1],
                        label=mod, alpha=0.6, s=10, color=colors[j])
        plt.title(title)
        plt.xlabel('t-SNE-1')
        plt.ylabel('t-SNE-2')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# --- Prepare Data ---
snr = 10
modulations = ['BPSK', 'QPSK', 'QAM16', 'QAM64']
Qmax = len(modulations)

with open('RML2016.10a_dict.pkl', 'rb') as f:
    data_dict = pickle.load(f, encoding='latin1')

X_all, Y_all = [], []
for idx, mod in enumerate(modulations):
    x = data_dict[(mod, snr)]
    iq = x[:, 0, :] + 1j * x[:, 1, :]
    X_all.extend(iq)
    Y_all.extend([idx] * len(iq))

X_all = np.array(X_all)
Y_all = np.array(Y_all)

idx_all = np.random.permutation(len(X_all))[:4000]
X_all = X_all[idx_all]
Y_all = Y_all[idx_all]

nTrain = int(0.7 * len(X_all))
nVal = int(0.1 * len(X_all))
nTest = len(X_all) - nTrain - nVal

X_train = X_all[:nTrain]
X_val = X_all[nTrain:nTrain+nVal]
X_test = X_all[nTrain+nVal:]
Y_train = Y_all[:nTrain]
Y_val = Y_all[nTrain:nTrain+nVal]
Y_test = Y_all[nTrain+nVal:]

Y_train_oh = np.eye(Qmax)[Y_train]
Y_val_oh = np.eye(Qmax)[Y_val]



latent_features = []
titles = ['Deep learning (MLP with absolute values of signal)', 'Deep learning (CNN with real and imaginary values of signal)',
          'Proposed method (with original signal)', 'Proposed method (with FFT-processed signal)', 'SCNN2']

# abs
X_train_abs = np.abs(X_train)
X_val_abs = np.abs(X_val)
X_test_abs = np.abs(X_test)
#model_abs = SimpleMLP2(128, 24, Qmax).to(device)
acc1, model_abs=train_eval_mlp(X_train_abs, X_val_abs, X_test_abs, Y_train_oh, Y_val_oh, Y_test, 128, 24, Qmax)
latent_features.append(extract_features(model_abs, X_test_abs))


# legacy realimag
X_train_realimag = get_real_imag_input(X_train)
X_val_realimag = get_real_imag_input(X_val)
X_test_realimag = get_real_imag_input(X_test)
acc2, model_realimag = train_eval_cnn2(X_train_realimag, X_val_realimag, X_test_realimag, Y_train, Y_val, Y_test, Qmax)
latent_features.append(extract_features(model_realimag, X_test_realimag))


# hankel
X_train_hk = np.array([make_hankel_onlyS(x) for x in X_train])
X_val_hk = np.array([make_hankel_onlyS(x) for x in X_val])
X_test_hk = np.array([make_hankel_onlyS(x) for x in X_test])
#model_hk = SimpleMLP2(64, 24, Qmax).to(device)
acc3, model_hk=train_eval_mlp(X_train_hk, X_val_hk, X_test_hk, Y_train_oh, Y_val_oh, Y_test, 64, 24, Qmax)
latent_features.append(extract_features(model_hk, X_test_hk))

# hankel-fft
X_train_hkf = np.array([make_hankel_onlyS(fft(x)) for x in X_train])
X_val_hkf = np.array([make_hankel_onlyS(fft(x)) for x in X_val])
X_test_hkf = np.array([make_hankel_onlyS(fft(x)) for x in X_test])
#model_hkf = SimpleMLP2(64, 24, Qmax).to(device)
acc4, model_hkf=train_eval_mlp2(X_train_hkf, X_val_hkf, X_test_hkf, Y_train_oh, Y_val_oh, Y_test, 64, 24, Qmax)
latent_features.append(extract_features(model_hkf, X_test_hkf))

# spectrogram
X_train_spec = np.array([make_spectrogram(x) for x in X_train])
X_val_spec = np.array([make_spectrogram(x) for x in X_val])
X_test_spec = np.array([make_spectrogram(x) for x in X_test])
#model_spec = TableICNN(Qmax).to(device)
acc5, model_spec=train_eval_cnn(X_train_spec, X_val_spec, X_test_spec, Y_train, Y_val, Y_test, Qmax)
latent_features.append(extract_features(model_spec, X_test_spec))

# --- t-SNE Visualization ---
visualize_tsne(latent_features, Y_test, modulations, titles)
