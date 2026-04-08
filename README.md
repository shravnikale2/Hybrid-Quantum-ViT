# Hybrid-Quantum-ViT
Data-efficient image classification using a hybrid quantum-classical Vision Transformer
# Hybrid Quantum-Classical Vision Transformer

## 📌 Project Overview
This project explores a hybrid quantum-classical approach for data-efficient image classification using a lightweight Vision Transformer (ViT) combined with a quantum feature augmentation layer.

The goal is to evaluate whether quantum feature transformations can improve learning performance under limited-data conditions.

---

## 🎯 Objective
- Design a Vision Transformer from scratch
- Integrate a quantum feature augmentation layer
- Compare classical and hybrid models
- Evaluate performance on small datasets (100, 500, 1000 samples)

---

## 🧠 Model Architecture

### Classical Model
- Lightweight Vision Transformer (ViT)
- Patch embedding + self-attention
- CLS token for classification

### Hybrid Model
- Vision Transformer backbone
- Quantum feature augmentation branch:
  - Feature reduction (128 → 4)
  - Quantum circuit using PennyLane
  - RX, RY rotations + CNOT entanglement
- Feature concatenation (128 + 4)
- Final classification layer

---

## ⚛️ Quantum Layer Details
- Number of qubits: 4  
- Encoding: Angle encoding (RY gates)  
- Trainable parameters: RX rotations  
- Entanglement: CNOT gates  
- Output: Expectation values of Pauli-Z operators  

---

## 📊 Dataset
- **Fashion-MNIST**
- 10 classes (clothing categories)
- Image size: 28×28 (resized to 32×32)

### Subsets Used
- 100 samples
- 500 samples
- 1000 samples

---

## 📈 Results

| Samples | Classical ViT | Hybrid ViT |
|--------|--------------|------------|
| 100    |  32.58%      | 36.27%     |
| 500    |  68.17%      |  69.23%    |
| 1000   |  69.90%      |  72.60%    |

---

## 📉 Key Observations

- Performance improves with increasing dataset size
- Hybrid model achieves performance comparable to classical model
- Naive quantum replacement failed due to gradient instability
- Augmentation-based hybrid design ensures stable learning
- Quantum layer acts as a feature enhancement module

---

## ⚠️ Challenges Faced

- Gradient instability (barren plateau problem)
- Information loss due to excessive feature compression
- Quantum simulation overhead

---

## 💡 Key Insight

> Quantum layers are more effective as feature augmentation modules rather than direct replacements in deep learning architectures.

---

## 🛠️ Tech Stack

- Python
- PyTorch
- PennyLane
- Torchvision
- Matplotlib

---

## 📁 Project Structure
Hybrid-Quantum-ViT/
├── models/ # ViT and hybrid model definitions
├── quantum/ # Quantum circuit implementation
├── experiments/ # Training and evaluation scripts
├── results/ # Graphs and result files
├── notebooks/ # Jupyter notebooks
├── README.md
├── .gitignore
---

## ▶️ How to Run

### 1. Install dependencies

pip install torch torchvision pennylane matplotlib


### 2. Run experiments

python experiments/run_experiments.py


---

## 📊 Output

- Accuracy comparison table
- Graph: Accuracy vs Dataset Size
- CSV file with results

---

## 🧾 Conclusion

The hybrid quantum-classical Vision Transformer demonstrates stable and comparable performance to the classical model under limited-data conditions. While quantum augmentation does not significantly outperform classical methods, it highlights the feasibility of hybrid architectures and the importance of proper integration design.

---

## 🔮 Future Work

- Test on larger datasets (CIFAR-10)
- Explore deeper quantum circuits
- Optimize quantum feature encoding
- Evaluate performance on real quantum hardware

---

## 👩‍💻 Authors

- Shravni Kale  
- Divya Thawkar
- Divyansh Hatwar

---

## 📌 Note

The `venv/` folder is excluded to keep the repository lightweight and within GitHub size lim
