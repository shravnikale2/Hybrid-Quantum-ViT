import torch
import torch.nn as nn
import pennylane as qml

from models.small_vit import SmallViT

# Quantum setup
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):

    # Encoding
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    # Trainable layer
    for i in range(n_qubits):
        qml.RX(weights[i], wires=i)

    # Entanglement
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_qubits))

    def forward(self, x):
        outputs = []

        for sample in x:
            q_out = quantum_circuit(sample, self.weights)
            q_out = torch.stack(q_out).float()
            outputs.append(q_out)

        return torch.stack(outputs)


class HybridViT_Augmented(nn.Module):
    def __init__(self):
        super().__init__()

        self.vit = SmallViT()

        # Reduce features for quantum input
        self.quantum_reduce = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, n_qubits)
        )

        self.quantum_layer = QuantumLayer()

        self.classifier = nn.Linear(128 + n_qubits, 10)

    def forward(self, x):

        # ===== Transformer =====
        x = self.vit.patch_embed(x)

        B = x.size(0)
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.vit.pos_embed
        x = self.vit.transformer(x)

        cls_output = x[:, 0]

        # ===== Quantum Branch =====
        q_input = self.quantum_reduce(cls_output)
        q_input = torch.tanh(q_input)

        q_output = self.quantum_layer(q_input)

        # ===== Combine =====
        combined = torch.cat([cls_output, q_output], dim=1)

        return self.classifier(combined)