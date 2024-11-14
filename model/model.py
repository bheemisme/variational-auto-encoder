from pathlib import Path
import torch
import torchvision.io as io
import torchvision.transforms as T

from model.VAE import VAE

# defining hyper-parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './checkpoints/vae_model_train_27_epoch_100.pth'
latent_dim=16
in_channels = 3


# loading the model
vae = VAE(in_channels=in_channels, latent_dim=latent_dim)
vae.load_state_dict(torch.load(model_path,
                               map_location=device,
                               weights_only=False))
vae.to(device=device)
vae.eval()

# compression function
def compress(image, file_name):
  image = image.to(device).type(torch.float)
  image = image.unsqueeze(0)

  with torch.no_grad():
    m, l = vae.encode(image)
    z = vae.reparameterize(m, l)

  z = z.cpu()
  z = z.squeeze(0)

  # serializing the compressed image
  compressed_file_path =  f'./temp/{file_name}.pt'
  torch.save(z, compressed_file_path)

  return compressed_file_path

def decompress(uploaded_file):
    z = torch.load(uploaded_file, weights_only=True)

    z = z.to(device).type(torch.float)
    z = z.unsqueeze(0)
    with torch.no_grad():
      image = vae.decode(z)

    image = image.cpu()
    image = image.type(torch.uint8)
    image = image.squeeze(0)

    file_path = './out-images/' + uploaded_file.name.split('.')[0] + '.jpg'
    io.write_jpeg(image, filename=file_path)

    return file_path

def preprocess_image(image_path: Path) -> torch.Tensor:
    img = io.read_image(str(image_path))
    if img.shape[0] == 1:
        img = img.repeat(3,1,1)
    transform = T.Resize((64,64))
    img = transform(img)
    img = img.to(device)
    return img

    



