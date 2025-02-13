from transformers import ViTForImageClassification, ViTFeatureExtractor
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Carregamento do conjunto de dados CIFAR-10
dataset = load_dataset("cifar10")

# Pré-processamento das imagens
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

def transform(examples):
    examples['pixel_values'] = feature_extractor(examples['img'], return_tensors="pt").pixel_values
    return examples

dataset = dataset.map(transform, batched=True)
dataset.set_format(type='torch', columns=['pixel_values', 'label'])

# Divisão do conjunto de dados em treino e teste
train_dataset = dataset['train']
test_dataset = dataset['test']

# Criação do DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Carregamento do modelo pré-treinado
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10,
    ignore_mismatched_sizes=True
)

# Definição do otimizador e da função de perda
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

# Treinamento do modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        inputs = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)

        outputs = model(inputs).logits
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item()}")

# Avaliação do modelo
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        inputs = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)

        outputs = model(inputs).logits
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%")

# Visualização de resultados
def show_image(img, label, predicted):
    img = img.permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.title(f"Label: {label}, Predicted: {predicted}")
    plt.show()

# Teste
for i in range(5):
    img = test_dataset[i]['pixel_values']
    label = test_dataset[i]['label']
    with torch.no_grad():
        output = model(img.unsqueeze(0).to(device)).logits
    _, predicted = torch.max(output, 1)
    show_image(img, label, predicted.item())