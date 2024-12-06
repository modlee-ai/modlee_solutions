import os
from torch import nn
from distilling_agentic_moderation import *


class TabularClassifier(modlee.model.TabularClassificationModleeModel):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 2*128),
            torch.nn.ReLU(),
            torch.nn.Linear(2*128, 2*64),
            torch.nn.ReLU(),
            torch.nn.Linear(2*64, num_classes)
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer

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
        texts, labels, test_size=0.2, random_state=42
    )

    # Datasets and DataLoaders
    train_dataset = ModerationDataset(train_texts, train_labels)
    val_dataset = ModerationDataset(val_texts, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4)

    # DNN Model

    # Input size is determined by the embedding size of the transformer model
    embedding_size = train_dataset.embeddings.shape[1]
    model = TabularClassifier(input_dim=embedding_size, num_classes=len(label_map))

    trainer = modlee.model.trainer.AutoTrainer(max_epochs=100)
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    # Count the trainable parameters in the sub-model
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Compute the number of trainable parameters
    trainable_params = count_parameters(model)
    print(f"Trainable parameters for custom end network: {trainable_params:,}")


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
