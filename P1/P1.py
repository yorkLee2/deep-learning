import numpy as np
from scipy.special import expit
from collections import Counter
from itertools import product

# ============== Helper Functions ==============
def generate_kmers(sequence, k):
    """Generate k-mers from a sequence."""
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def extract_kmer_features(sequences, k):
    """Extract k-mer features for all sequences."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    kmer_vocab = [''.join(kmer) for kmer in product(amino_acids, repeat=k)]
    kmer_to_index = {kmer: i for i, kmer in enumerate(kmer_vocab)}
    
    feature_matrix = np.zeros((len(sequences), len(kmer_vocab)))
    for i, seq in enumerate(sequences):
        kmers = generate_kmers(seq, k)
        kmer_counts = Counter(kmers)
        for kmer, count in kmer_counts.items():
            if kmer in kmer_to_index:
                feature_matrix[i, kmer_to_index[kmer]] = count

    # 
    feature_matrix /= (feature_matrix.sum(axis=1, keepdims=True) + 1e-6)
    return feature_matrix


# ============== Neural Network Class ==============
class AdvancedNeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, 
                 learning_rate=0.001, l2_reg=1e-5, dropout_rate=0.0):
        """
        : input -> hidden1 -> hidden2 -> output
        """
        # 
        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size1))

        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0 / hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))

        self.W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2.0 / hidden_size2)
        self.b3 = np.zeros((1, output_size))

        self.learning_rate = learning_rate
        self.l2_reg = l2_reg  # L2 
        self.dropout_rate = dropout_rate

        # Adam
        self.mW1, self.vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.mW2, self.vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.mW3, self.vW3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        self.mb1, self.vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.mb2, self.vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.mb3, self.vb3 = np.zeros_like(self.b3), np.zeros_like(self.b3)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0   

    def leaky_relu(self, z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)

    def leaky_relu_derivative(self, z, alpha=0.01):
        return np.where(z > 0, 1, alpha)

    def sigmoid(self, z):
        return expit(z)

    def dropout(self, A):
        """

        """
        if self.dropout_rate <= 0.0:
            # no dropout
            return A, np.ones_like(A)
        mask = (np.random.rand(*A.shape) > self.dropout_rate).astype(A.dtype)
        A_dropped = A * mask / (1.0 - self.dropout_rate)
        return A_dropped, mask

    def forward(self, X, training=True):
        """
      
        X -> (W1, b1) -> A1 -> dropout -> (W2, b2) -> A2 -> dropout -> (W3, b3) -> output
        """
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.leaky_relu(self.Z1)
        if training:
            self.A1, self.mask1 = self.dropout(self.A1)
        else:
            self.mask1 = np.ones_like(self.A1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.leaky_relu(self.Z2)
        if training:
            self.A2, self.mask2 = self.dropout(self.A2)
        else:
            self.mask2 = np.ones_like(self.A2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.sigmoid(self.Z3)
        self.A3 = np.clip(self.A3, 1e-6, 1 - 1e-6)  # prevent inf
        
        return self.A3

    def backward(self, X, y, class_weights):
        """
   
        """
        m = X.shape[0]
        # dZ3
        weights = np.array([class_weights[int(label)] for label in y])
        dZ3 = (self.A3 - y.reshape(-1, 1)) * weights.reshape(-1, 1)

        # dW3, db3
        dW3 = np.dot(self.A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        # backprop to A2
        dA2 = np.dot(dZ3, self.W3.T)
        # dropout mask
        dA2 *= self.mask2
        # dZ2
        dZ2 = dA2 * self.leaky_relu_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # backprop to A1
        dA1 = np.dot(dZ2, self.W2.T)
        dA1 *= self.mask1
        # dZ1
        dZ1 = dA1 * self.leaky_relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m


        dW3 += self.l2_reg * self.W3
        dW2 += self.l2_reg * self.W2
        dW1 += self.l2_reg * self.W1

      
        self.t += 1  
        # helper function
        def adam_update(param, dparam, mparam, vparam):
            mparam = self.beta1 * mparam + (1 - self.beta1) * dparam
            vparam = self.beta2 * vparam + (1 - self.beta2) * (dparam ** 2)

            m_hat = mparam / (1 - self.beta1 ** self.t)
            v_hat = vparam / (1 - self.beta2 ** self.t)

            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            return param, mparam, vparam

        # W3, b3
        self.W3, self.mW3, self.vW3 = adam_update(self.W3, dW3, self.mW3, self.vW3)
        self.b3, self.mb3, self.vb3 = adam_update(self.b3, db3, self.mb3, self.vb3)
        # W2, b2
        self.W2, self.mW2, self.vW2 = adam_update(self.W2, dW2, self.mW2, self.vW2)
        self.b2, self.mb2, self.vb2 = adam_update(self.b2, db2, self.mb2, self.vb2)
        # W1, b1
        self.W1, self.mW1, self.vW1 = adam_update(self.W1, dW1, self.mW1, self.vW1)
        self.b1, self.mb1, self.vb1 = adam_update(self.b1, db1, self.mb1, self.vb1)


    def compute_loss(self, y_true, y_pred):
        """
       
        y_true, y_pred: shape = (batch_size, 1)
        """
        m = y_true.shape[0]
        # 
        log_loss = -np.mean(y_true * np.log(y_pred + 1e-6) + (1 - y_true) * np.log(1 - y_pred + 1e-6))
        # L2 
        l2_loss = 0.5 * self.l2_reg * (np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.W3**2))
        return log_loss + l2_loss

    def train(self, X, y, 
              epochs=100, 
              batch_size=32, 
              class_weights=None, 
              X_val=None, 
              y_val=None):

        m = X.shape[0]
        if class_weights is None:
     
            class_weights = {0:1.0, 1:1.0}
        
        for epoch in range(epochs):
            
            indices = np.random.permutation(m)
            X_shuffled, y_shuffled = X[indices], y[indices]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # forward
                output = self.forward(X_batch, training=True)
                # backward
                self.backward(X_batch, y_batch, class_weights)

            train_pred = self.forward(X_shuffled, training=False)
            train_loss = self.compute_loss(y_shuffled, train_pred)
            
            if X_val is not None and y_val is not None:

                val_pred = self.forward(X_val, training=False)
                val_loss = self.compute_loss(y_val, val_pred)
                val_acc = np.mean((val_pred >= 0.5).astype(int) == y_val.reshape(-1,1))
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

    def predict(self, X):
        probabilities = self.forward(X, training=False)

        return np.where(probabilities >= 0.5, 1, 0)

# ============== Load Data ==============
train_file_path = "train.dat"
test_file_path = "test.dat"

with open(train_file_path, "r") as f:
    train_data = f.readlines()

with open(test_file_path, "r") as f:
    test_data = f.readlines()

train_labels = []
train_sequences = []
for line in train_data:
    label, sequence = line.strip().split("\t")
    train_labels.append(int(label))
    train_sequences.append(sequence)

test_sequences = [line.strip() for line in test_data]

# ============== Feature Extraction ==============
k = 2  # 
X_train = extract_kmer_features(train_sequences, k)
X_test = extract_kmer_features(test_sequences, k)

y_train = np.array(train_labels)

y_train = (y_train + 1) // 2  


split_ratio = 0.8
split_index = int(len(X_train) * split_ratio)

X_train_, X_val = X_train[:split_index], X_train[split_index:]
y_train_, y_val = y_train[:split_index], y_train[split_index:]


negative_count = np.sum(y_train_ == 0)
positive_count = np.sum(y_train_ == 1)
if positive_count == 0: 

    class_weights = {0:1.0, 1:1.0}
else:
    class_weights = {0:1.0, 1: negative_count / (positive_count + 1e-6)}


nn = AdvancedNeuralNetwork(input_size=X_train_.shape[1],
                           hidden_size1=256,
                           hidden_size2=128,
                           output_size=1,
                           learning_rate=0.001,     
                           l2_reg=1e-5,             
                           dropout_rate=0.2)       

nn.train(X_train_, y_train_,
         epochs=100,       
         batch_size=32,
         class_weights=class_weights,
         X_val=X_val,      
         y_val=y_val)


test_predictions = nn.predict(X_test)


test_predictions_transformed = np.where(test_predictions == 0, -1, 1)

with open("prediction.txt", "w") as f:
    for pred in test_predictions_transformed.flatten():
        f.write(f"{int(pred)}\n")
