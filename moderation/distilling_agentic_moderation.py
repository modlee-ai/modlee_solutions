import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import modlee

modlee.init(api_key=os.getenv("MODLEE_API_KEY"))
# Model name for embeddings
# transformer_model_name = "bert-base-uncased"
transformer_model_name = "huawei-noah/TinyBERT_General_4L_312D"
max_length=128

class ModerationDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        self.model = AutoModel.from_pretrained(transformer_model_name)
        self.max_length = max_length

        # Set the device for embeddings computation
        self.device = torch.device("cpu")

        # Precompute embeddings for all texts
        self.embeddings = self._compute_embeddings()

    def _compute_embeddings(self):
        """Compute embeddings for all texts using the transformer model."""
        self.model.eval()
        self.model.to(self.device)
        embeddings = []
        with torch.no_grad():
            for text in self.texts:
                tokens = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                tokens = {key: val.to(self.device) for key, val in tokens.items()}
                outputs = self.model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                )
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
                embeddings.append(cls_embedding.squeeze(0))
        return torch.stack(embeddings)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], torch.tensor(self.labels[idx], dtype=torch.long)


def load_data(data_dir):
    texts = []
    labels = []
    label_map = {}

    for filename in os.listdir(data_dir):
        if filename.endswith(".json") and 'distilled_model' not in filename:
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r") as file:
                data = json.load(file)
                label = data["output"]
                
                # Dynamically assign a numeric label if not already mapped
                if label not in label_map:
                    label_map[label] = len(label_map)
                
                texts.append(data["input"])
                labels.append(label_map[label])

    print(f"Generated label map: {label_map}")
    return texts, labels, label_map


def load_model_for_inference(model_path):
    # Load the model
    device = torch.device("cpu")
    model = torch.jit.load(model_path)
    model.eval()
    return model,device

def run_inference_with_model(model,device,input_text):
    """
    Load a PyTorch model, preprocess input text, and run inference.

    Args:
        model_path (str): Path to the PyTorch model file (.pth).
        input_text (str): The input text to be classified.
        transformer_model_name (str): Name of the transformer model used for tokenization.
        max_length (int): Maximum length for tokenized input.

    Returns:
        int: The predicted class index from the model.
    """

    inference_dataset = ModerationDataset([input_text], [0])
    inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=True)

    with torch.no_grad():
        for embeddings, labels in inference_dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs, 1)

    return predicted


def main():

    device = torch.device("cpu")  # Explicitly set to CPU

    # Path to the directory containing JSON files
    data_dir = "./moderation_results"
    model_save_path = os.path.join(data_dir, "distilled_model.pth")
    details_save_path = os.path.join(data_dir, "distilled_model_details.json")

    # Load data
    texts, labels, label_map = load_data(data_dir)

    # Print the label map
    print(f"Label map: {label_map}")

    # Train-test split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.5, random_state=42
    )

    # Datasets and DataLoaders
    train_dataset = ModerationDataset(train_texts, train_labels)
    val_dataset = ModerationDataset(val_texts, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4)

    # DNN Model
    embedding_size = train_dataset.embeddings.shape[1]
    recommender = modlee.recommender.TabularClassificationRecommender(num_classes=len(label_map))
    recommender.fit(train_dataloader)
    model = recommender.model.to(device)
    print(f"\nRecommended model: \n{model}")

    trainer = modlee.model.trainer.AutoTrainer(max_epochs=100)
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    # Evaluate validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for embeddings, labels in val_dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_accuracy = (correct / total) * 100
    print(f"Validation Accuracy: {final_accuracy:.2f}%")

    # Save model and details conditionally
    if os.path.exists(details_save_path):
        with open(details_save_path, "r") as f:
            existing_details = json.load(f)
        existing_accuracy = existing_details.get("final_accuracy_%", 0)
    else:
        existing_accuracy = 0

    if final_accuracy > existing_accuracy:
        # Convert to TorchScript using tracing
        dummy_input = torch.randn(1, embedding_size)  # Single example with correct embedding size
        traced_model = torch.jit.trace(model, dummy_input)
        # Save the TorchScript model
        traced_model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

        # Save model details to JSON
        details_data = {
            "final_accuracy_%": final_accuracy,
            "label_map": label_map
        }
        with open(details_save_path, "w") as f:
            json.dump(details_data, f, indent=4)
        print(f"Model details saved to {details_save_path}")
    else:
        print(f"Model not saved. Current accuracy ({final_accuracy}%) is not higher than existing accuracy ({existing_accuracy}%).")



if __name__ == "__main__":
    main()
