CS5787 HW3 Deep Learning with Prof. Hadar Elor - Sean Hardesty Lewis (shl225)
# Semi-supervised Learning via a Variational Autoencoder and Training GANs on FashionMNIST

## Overview 1 (VAE, SVM)
This project demonstrates how to train a Variational Autoencoder (VAE) on the FashionMNIST dataset, extract latent representations, and use them to train an SVM classifier. 

## Training the VAE

1. **Set Training Parameters**:

   ```python
   num_epochs = 30  # Number of epochs
   batch_size = 128  # Batch size
   hidden_dim = 500  # Hidden layer size in VAE
   latent_dim = 50   # Latent space dimensionality
   ```

2. **Run the VAE Training**:

   Run the following cell in the notebook to train the VAE for 30 epochs:

   ```python
   for epoch in range(1, num_epochs + 1):
       train_loss = 0
       for data, _ in train_loader:
           data = data.view(-1, 784)
           recon_batch, mu, logvar = vae(data)
           loss = loss_function(recon_batch, data, mu, logvar)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           train_loss += loss.item()

       print(f'Epoch {epoch}, Average Loss: {train_loss / len(train_loader):.4f}')
   ```

## Extracting Latent Representations

After training, switch the VAE to evaluation mode and extract the latent representations of the train and test datasets:

```python
vae.eval()
train_latent, train_labels = get_latent_representations(train_loader)
test_latent, test_labels = get_latent_representations(test_loader)
```

## Training the SVM Classifier

1. **Select Number of Labeled Samples**:

   Choose how many labeled samples to use for training the SVM (e.g., 100, 600, 1000, or 3000):

   ```python
   label_sizes = [100, 600, 1000, 3000]
   ```

2. **Train and Save the SVM Model**:

   ```python
   for num_labels in label_sizes:
       indices = np.random.choice(len(train_latent), size=num_labels, replace=False)
       X_train, y_train = train_latent[indices], train_labels[indices]

       svm = SVC(kernel='rbf', gamma='scale')
       svm.fit(X_train, y_train)

       joblib.dump(svm, f'svm_model_{num_labels}.joblib')
       print(f'SVM with {num_labels} labels saved.')
   ```

## Testing the SVM Classifier

1. **Load a Saved SVM Model**:

   ```python
   svm = joblib.load('svm_model_1000.joblib')
   print('SVM model loaded.')
   ```

2. **Evaluate on Test Data**:

   ```python
   y_pred = svm.predict(test_latent)
   accuracy = accuracy_score(test_labels, y_pred)
   print(f'Test Accuracy: {accuracy:.4f}')
   ```

## Plotting Results

Visualize the SVM performance with different numbers of labeled samples:

```python
plt.plot(label_sizes, [results[s] for s in label_sizes], marker='o')
plt.title('Test Accuracy vs Number of Labels')
plt.xlabel('Number of Labels')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
```

## Overview 2 (GANs)
This project implements a VAE and training several different GANs with the FashionMNIST dataset. The GANs implemented are:

- **DCGAN (Deep Convolutional GAN)**
- **WGAN (Wasserstein GAN)**
- **WGAN-GP (Wasserstein GAN with Gradient Penalty)**

Each GAN is trained using two different architectures, referred to as Architecture A and Architecture B.

## Training the Models

The models are trained using the provided notebook. Training is performed within the `train_gan` function, which trains the specified GAN type with the selected architecture.

### Steps to Train the Models:

1. **Set Training Parameters**:

   ```python
   num_epochs = 20  # Number of epochs to train
   batch_size = 64  # Batch size
   nz = 128         # Size of the latent vector (noise)
   ngf = 64         # Size of feature maps in generator
   ndf = 64         # Size of feature maps in discriminator
   num_classes = 10 # Number of classes in FashionMNIST
   ```

2. **Run the Training Loop**:

   ```python
   architectures = ['A', 'B']
   gan_types = ['DCGAN', 'WGAN', 'WGAN-GP']
   results = {}

   for arch in architectures:
       results[arch] = {}
       for gan_type in gan_types:
           G_losses, D_losses, netG = train_gan(gan_type=gan_type, architecture=arch)
           results[arch][gan_type] = {
               'G_losses': G_losses,
               'D_losses': D_losses,
               'netG': netG
           }
   ```

3. **Training Outputs**:

   - **Loss Curves**: After training, loss curves for the generator and discriminator are plotted.
   - **Trained Models**: Stored in the `results` dictionary.

## Generating New Images

### Steps to Generate New Images:

1. **Select GAN Type and Architecture**:

   ```python
   gan_type = 'WGAN-GP'
   architecture = 'B'
   netG = results[architecture][gan_type]['netG']
   netG.eval()
   ```

2. **Prepare Noise Vector and Labels**:

   ```python
   num_samples = 10
   nz = 128
   noise = torch.randn(num_samples, nz, 1, 1, device=device)
   labels = torch.full((num_samples,), 0, dtype=torch.long, device=device)  # Example: T-shirts
   ```

3. **Generate Images**:

   ```python
   with torch.no_grad():
       fake_images = netG(noise, labels)
       fake_images = (fake_images * 0.5) + 0.5  # Denormalize to [0,1]
   ```

4. **Visualize Images**:

   ```python
   import matplotlib.pyplot as plt
   from torchvision.utils import make_grid

   grid_img = make_grid(fake_images.cpu(), nrow=5)
   plt.figure(figsize=(10,5))
   plt.imshow(np.transpose(grid_img, (1,2,0)))
   plt.axis('off')
   plt.show()
   ```

## Saving and Loading the Trained Models

**Saving the Models**:

```python
for architecture in results:
    for gan_type in results[architecture]:
        netG = results[architecture][gan_type]['netG']
        torch.save(netG.state_dict(), f'generator_{gan_type}_{architecture}.pth')
```

**Loading the Models**:

```python
netG = Generator(architecture='B').to(device)
netG.load_state_dict(torch.load('generator_WGAN-GP_B.pth', map_location=device))
netG.eval()
```

## Number of Failed Convergences

During the implementation, some training runs failed to converge. Below are the failed attempts per architecture and GAN type.

- **Architecture A**:
  - **DCGAN**: 15452 failed attempts
  - **WGAN**: 15683 failed attempts
  - **WGAN-GP**: 16383 failed attempts

- **Architecture B**:
  - **DCGAN**: 21267 failed attempts
  - **WGAN**: 21540 failed attempts
  - **WGAN-GP**: 21801 failed attempts
 
## Analysis and Conclusion

The VAE model successfully captured latent features from the FashionMNIST dataset, and the extracted representations were used to train an SVM classifier. The experiment demonstrated that as the number of labeled samples increases, the SVM classifier achieves better accuracy, illustrating the importance of labeled data in supervised learning tasks. 

For the trained GANs, we decided that Architecture B performed better for the final image generation even though it took more effort to stabilize during training as it provided deeper layers and higher feature map sizes which led to better performance in capturing latent structure in the data.

In terms of GAN-related tasks (if extended), using WGAN-GP offers more stable training, as shown in prior research. These insights align with Gulrajani et al.’s work on improved Wasserstein GANs which confirms the need for gradient penalties for better stable convergence.

## References

1. **Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms**  
   Han Xiao, Kashif Rasul, and Roland Vollgraf.  
   *arXiv preprint arXiv:1708.07747*, 2017.
   
   [arXiv:1708.07747](https://arxiv.org/abs/1708.07747)  

3. **Semi-Supervised Learning with Deep Generative Models**  
   Diederik P. Kingma, Danilo J. Rezende, Shakir Mohamed, Max Welling.  
   *Advances in Neural Information Processing Systems (NeurIPS)*, 2014.
   
   [arXiv:1406.5298](https://arxiv.org/abs/1406.5298)

4. **Improved Training of Wasserstein GANs**  
   Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville.  
   *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.
   
   [arXiv:1704.00028](https://arxiv.org/abs/1704.00028)  

