import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef

# load
def load_data(file_path):
    sequences, labels = [], []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                labels.append(int(parts[0]))
                sequences.append(parts[1])
    return sequences, labels

train_sequences, train_labels = load_data("train.dat")
test_sequences = [line.strip() for line in open("test.dat", "r").readlines()]

# k=3
def extract_kmer_features(sequences, all_kmers=None, k=3):
    kmer_counts = []
    for seq in sequences:
        kmer_dict = {}
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if kmer in kmer_dict:
                kmer_dict[kmer] += 1
            else:
                kmer_dict[kmer] = 1
        kmer_counts.append(kmer_dict)

   
    if all_kmers is None:
        all_kmers = sorted(set(k for d in kmer_counts for k in d))

   
    feature_matrix = torch.zeros((len(sequences), len(all_kmers)))
    for i, kmer_dict in enumerate(kmer_counts):
        for j, kmer in enumerate(all_kmers):
            feature_matrix[i, j] = kmer_dict.get(kmer, 0)

    return feature_matrix, all_kmers


X_train, kmer_vocab = extract_kmer_features(train_sequences, k=3)

X_test, _ = extract_kmer_features(test_sequences, all_kmers=kmer_vocab, k=3)


scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)  
y_train = torch.tensor(train_labels, dtype=torch.float32)


y_train = (y_train + 1) / 2
y_val = (y_train + 1) / 2  


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)


class PeptideDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

train_dataset = PeptideDataset(X_train, y_train)
val_dataset = PeptideDataset(X_val, y_val)
test_dataset = PeptideDataset(X_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class PeptideClassifier(nn.Module):
    def __init__(self, input_dim):
        super(PeptideClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PeptideClassifier(X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)  

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # labels  [0,1] 
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # MCC
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs).cpu().numpy().flatten()
            preds = np.where(outputs > 0.5, 1, -1)
            val_preds.extend(preds)
            val_targets.extend(labels.cpu().numpy())

    mcc_score = matthews_corrcoef(val_targets, val_preds)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, MCC: {mcc_score:.4f}")


model.eval()
test_preds = []
with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs).cpu().numpy().flatten()
        preds = np.where(outputs > 0.5, 1, -1)
        test_preds.extend(preds)


output_filename = "prediction.txt"
np.savetxt(output_filename, test_preds, fmt="%d")

print(f"✅ 预测完成，结果已保存为 {output_filename}")
