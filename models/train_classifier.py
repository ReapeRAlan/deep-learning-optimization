def train_classifier(classifier, train_latent, train_labels, criterion, optimizer, device, num_epochs=100):
    classifier.train()
    for epoch in range(num_epochs):
        running_loss = 0
        for latent, labels in zip(train_latent, train_labels):
            latent, labels = latent.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = classifier(latent)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Época {epoch+1}/{num_epochs} - Pérdida Clasificador: {running_loss:.4f}")
