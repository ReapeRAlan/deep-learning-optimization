def train_autoencoder(autoencoder, data_loader, criterion, optimizer, device, num_epochs=50):
    autoencoder.train()
    for epoch in range(num_epochs):
        running_loss = 0
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()

            # Forward pass
            reconstructed, _ = autoencoder(inputs)
            loss = criterion(reconstructed, inputs)

            # Backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Época {epoch+1}/{num_epochs} - Pérdida Autoencoder: {running_loss / len(data_loader):.4f}")
