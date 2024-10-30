!pip install torch_snippets torch_summary --quiet
import itertools
from PIL import Image
from torch_snippets import *
from torchvision import transforms
from torchvision.utils import make_grid
from torchsummary import summary

import os
import matplotlib.pyplot as plt
# Display an image from each folder
subfolders = ['trainA', 'trainB', 'testA', 'testB']
dataset_path = './extracted_dataset/HW_06 Dataset'

for subfolder in subfolders:
    folder_path = os.path.join(dataset_path, subfolder)
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)

        if len(files) > 0:
            # Display the first image in the folder
            img_path = os.path.join(folder_path, files[0])
            img = Image.open(img_path)

            # Plot the image using matplotlib
            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.title(f"Image from {subfolder} folder: {files[0]}")
            plt.axis('off')
            plt.show()
        else:
            print(f"\n{subfolder} folder is empty.")
    else:
        print(f"\nError: Path '{folder_path}' does not exist.")

IMAGE_SIZE = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE*1.33)),
    transforms.RandomCrop((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class Report:
    """Utility to keep track of the training progress"""
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs
        self.records = []

    def record(self, epoch, **kwargs):
        # Ensure all logged values are numeric
        numeric_kwargs = {k: float(v) for k, v in kwargs.items() if isinstance(v, (int, float))}
        self.records.append((epoch, numeric_kwargs))
        print(f"Epoch {epoch:.2f}: {numeric_kwargs}", end='\r')

    def report_avgs(self, epoch):
        filtered_records = [d for e, d in self.records if int(e) == epoch]
        if not filtered_records:
            print(f"No records found for epoch {epoch}")
            return
        avgs = {k: sum(d[k] for d in filtered_records) / len(filtered_records) for k in filtered_records[0]}
        print(f"Epoch {epoch} averages: {avgs}")

class Monet2PhotoDataset(Dataset):
    def __init__(self, monet, photo):
        # Get list of images
        self.monet = Glob(os.path.join(dataset_path, monet, '*.jpg'))
        self.photo = Glob(os.path.join(dataset_path, photo, '*.jpg'))
        self.transform = transform

        print(f"Monet images found: {len(self.monet)}")
        print(f"Photo images found: {len(self.photo)}")

        if len(self.monet) == 0 or len(self.photo) == 0:
            raise ValueError("Both Monet and Photo datasets must contain at least one image.")

    def __getitem__(self, ix):
        # Choose one Monet image and one random Photo image
        monet_path = self.monet[ix % len(self.monet)]
        photo_path = self.choose()

        monet = Image.open(monet_path).convert('RGB')
        photo = Image.open(photo_path).convert('RGB')

        # Apply transformation if defined
        if self.transform:
            monet = self.transform(monet)
            photo = self.transform(photo)

        return monet, photo

    def __len__(self):
        return max(len(self.monet), len(self.photo))

    def choose(self):
        return self.photo[randint(0, len(self.photo) - 1)]

    def collate_fn(self, batch):
        # Collate function to create a batch of images
        srcs, trgs = list(zip(*batch))
        srcs = torch.cat([img[None] for img in srcs], 0).to(device).float()
        trgs = torch.cat([img[None] for img in trgs], 0).to(device).float()
        return srcs, trgs

# Create Training and Validation Datasets
trn_ds = Monet2Photo('trainA', 'trainB')
val_ds = Monet2Photo('testA', 'testB')

# Create DataLoader for Training and Validation Sets
trn_dl = DataLoader(trn_ds, batch_size=1, shuffle=True, collate_fn=trn_ds.collate_fn)
val_dl = DataLoader(val_ds, batch_size=5, shuffle=True, collate_fn=val_ds.collate_fn)

# Test DataLoader Iteration
for batch in trn_dl:
    monet_batch, photo_batch = batch
    print(f"Monet batch shape: {monet_batch.shape}, Photo batch shape: {photo_batch.shape}")
    break

#Weight intialization and Residual block
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )
    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()
        out_features = 64
        channels = 3
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features
        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)
        self.apply(weights_init_normal)
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        channels, height, width = 3, IMAGE_SIZE, IMAGE_SIZE

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )
        self.apply(weights_init_normal)

    def forward(self, img):
        return self.model(img)

# Instantiate the models
generator = GeneratorResNet(num_residual_blocks=9)
discriminator = Discriminator()

# Apply weights initialization (already applied within the constructor)
# Move models to the device
generator.to(device)
discriminator.to(device)

# Example optimizer definitions
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

@torch.no_grad()
def generate_sample():
    # Get a batch of validation data
    data = next(iter(val_dl))
    G_AB.eval()
    G_BA.eval()

    real_A, real_B = data

    # Move data to the same device as the model
    real_A, real_B = real_A.to(device), real_B.to(device)

    # Generate fake images
    fake_B = G_AB(real_A)
    fake_A = G_BA(real_B)

    # Arrange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)

    # Arrange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)

    # Show the grid of images
    show(image_grid.detach().cpu().permute(1, 2, 0).numpy(), sz=12)

# Generator Training Step
def train_generator_step(G_AB, G_BA, D_B, optimizer_G, real_A, real_B, criterion_GAN, lambda_cycle, lambda_identity):
    optimizer_G.zero_grad()

    # GAN loss for G_AB (A -> B) and G_BA (B -> A)
    fake_B = G_AB(real_A)
    pred_fake = D_B(fake_B)
    loss_GAN_AB = criterion_GAN(pred_fake, torch.ones_like(pred_fake))

    fake_A = G_BA(real_B)
    pred_fake = D_A(fake_A)
    loss_GAN_BA = criterion_GAN(pred_fake, torch.ones_like(pred_fake))

    # Cycle consistency loss
    recovered_A = G_BA(fake_B)
    loss_cycle_A = torch.nn.functional.l1_loss(recovered_A, real_A) * lambda_cycle

    recovered_B = G_AB(fake_A)
    loss_cycle_B = torch.nn.functional.l1_loss(recovered_B, real_B) * lambda_cycle

    # Identity loss
    loss_identity_A = torch.nn.functional.l1_loss(G_BA(real_A), real_A) * lambda_identity
    loss_identity_B = torch.nn.functional.l1_loss(G_AB(real_B), real_B) * lambda_identity

    # Total generator loss
    loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_identity_A + loss_identity_B
    loss_G.backward()
    optimizer_G.step()

    return loss_G.item()

# Discriminator Training Step
def train_discriminator_step(D, optimizer_D, real_imgs, fake_imgs, criterion_GAN):
    optimizer_D.zero_grad()

    # Real loss
    pred_real = D(real_imgs)
    loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

    # Fake loss
    pred_fake = D(fake_imgs.detach())
    loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

    # Total loss
    loss_D = (loss_real + loss_fake) * 0.5
    loss_D.backward()
    optimizer_D.step()

    return loss_D.item()

G_AB = GeneratorResNet().to(device)
G_BA = GeneratorResNet().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

lambda_cyc, lambda_id = 10.0, 5.0

n_epochs = 20
log = Report(n_epochs)
for epoch in range(n_epochs):
    G_AB.train()
    G_BA.train()
    D_A.train()
    D_B.train()

    N = len(trn_dl)
    for bx, batch in enumerate(trn_dl):
        real_A, real_B = batch
        real_A, real_B = real_A.to(device), real_B.to(device)

        # Train generators
        loss_G = train_generator_step(G_AB, G_BA, D_B, optimizer_G, real_A, real_B, criterion_GAN, lambda_cyc, lambda_id)

        # Generate fake images
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)

        # Train discriminators
        loss_D_A = train_discriminator_step(D_A, optimizer_D_A, real_A, fake_A, criterion_GAN)
        loss_D_B = train_discriminator_step(D_B, optimizer_D_B, real_B, fake_B, criterion_GAN)
        loss_D = (loss_D_A + loss_D_B) / 2

        # Log losses
        log.record(epoch + (1 + bx) / N, loss_D=loss_D, loss_G=loss_G, end='\r')

    # Generate sample images after each epoch
    generate_sample()

    # Report average losses for the epoch
    log.report_avgs(epoch + 1)

import matplotlib.pyplot as plt

# Plotting the losses over epochs
g_losses = [5.574812889099121, 5.066128730773926, 6.5337395668029785, 6.662298679351807, 5.237408638000488,
            6.675480842590332, 8.176063537597656, 7.0415358543396, 6.052495002746582, 4.198227882385254,
            5.00065279006958, 6.451258659362793, 6.166991710662842, 6.7870588302612305, 5.712403774261475,
            6.057143211364746, 5.302469253540039, 5.332411766052246, 5.222959995269775, 7.361080169677734]

d_losses = [0.3521578907966614, 0.2403830736875534, 0.22746148705482483, 0.12169665098190308, 0.10385331884026527,
            0.07161591947078705, 0.043687108904123306, 0.21296221762895584, 0.07078397087752819, 0.0965602807700634,
            0.1084199845790863, 0.08818954043090343, 0.08836795017123222, 0.10674545913934708, 0.15370044857263565,
            0.05648935027420521, 0.13360285386443138, 0.046843934804201126, 0.07676876708865166, 0.16625919193029404]

# Plotting the losses over epochs
epochs = range(1, len(g_losses) + 1)

plt.figure(figsize=(10, 6))

# Plot generator loss
plt.plot(epochs, g_losses, label='Generator Loss', color='blue')

# Plot discriminator loss
plt.plot(epochs, d_losses, label='Discriminator Loss', color='red')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Generator and Discriminator Losses Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
