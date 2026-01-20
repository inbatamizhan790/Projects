# fsp_aoi_full_pipeline_vis_overlay_updated.py
import os, random, math, pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import cv2
import numpy as np

# ---------------------------
# Synthetic dataset generator3

# ---------------------------
def create_ellipse_image(a_px:int, b_px:int, angle_deg:float, img_size=(256,256), center=None):
    img = Image.new('L', img_size, color=255)
    base = Image.new('L', img_size, color=0)
    draw = ImageDraw.Draw(base)
    if center is None:
        center = (img_size[0]//2, img_size[1]//2)
    x0 = center[0]-a_px; y0=center[1]-b_px
    x1 = center[0]+a_px; y1=center[1]+b_px
    draw.ellipse([x0,y0,x1,y1], fill=200)
    base = base.rotate(angle_deg, center=center, resample=Image.BICUBIC, expand=False)
    img = Image.composite(base,img,base)
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5,2.0)))
    draw2 = ImageDraw.Draw(img)
    for _ in range(random.randint(0,6)): 
        rx = center[0]+random.randint(-a_px*2,a_px*2)
        ry = center[1]+random.randint(-b_px*2,b_px*2)
        r=random.randint(1,4)
        draw2.ellipse([rx-r,ry-r,rx+r,ry+r],fill=random.randint(50,200))
    return img

def generate_synthetic_dataset(outdir='synthetic_dataset', n=500, img_size=(256,256)):
    os.makedirs(outdir, exist_ok=True)
    rows=[]
    for i in range(n):
        a=random.randint(20,80)
        theta=random.uniform(5.0,75.0)
        b=max(1,int(a*math.sin(math.radians(theta))))
        ang=random.uniform(0.0,180.0)
        img=create_ellipse_image(a,b,ang,img_size)
        fname=f'stain_{i:04d}.png'
        img.save(os.path.join(outdir,fname))
        rows.append({'filename':fname,'aoi_deg':theta})
    df=pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir,'labels.csv'),index=False)
    print(f"Synthetic dataset {n} images saved to {outdir}")

# ---------------------------
# PyTorch Dataset & Model
# ---------------------------
class StainDatasetTorch(Dataset):
    def __init__(self,csv_file,img_dir,transform=None,label_scale=90.0):
        self.df=pd.read_csv(csv_file)
        self.img_dir=img_dir
        self.transform=transform
        self.label_scale=label_scale
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        row=self.df.iloc[idx]
        img_path=os.path.join(self.img_dir,row['filename'])
        img=Image.open(img_path).convert('RGB')
        if self.transform:
            img=self.transform(img)
        label=float(row['aoi_deg'])/self.label_scale
        return img, torch.tensor([label],dtype=torch.float32)

def get_resnet_regression(pretrained=True, freeze_backbone=False):
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    return model

# ---------------------------
# Train function with visualization
# ---------------------------
def train_resnet(csv_train,csv_val,img_dir,epochs=20,batch_size=16,lr=1e-4,label_scale=90.0,save_path='resnet_aoi.pth'):
    transform_train=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224,scale=(0.8,1.0)),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    transform_val=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_ds=StainDatasetTorch(csv_train,img_dir,transform_train,label_scale)
    val_ds=StainDatasetTorch(csv_val,img_dir,transform_val,label_scale)
    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=2)
    val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=2)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=get_resnet_regression(pretrained=True)
    model=model.to(device)
    optimizer=optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=lr)
    criterion=nn.MSELoss()
    best_val=1e9
    train_losses=[]
    val_losses=[]
    for epoch in range(epochs):
        model.train()
        running=0.0
        for imgs,labels in train_loader:
            imgs,labels=imgs.to(device),labels.to(device)
            optimizer.zero_grad()
            outs=model(imgs)
            loss=criterion(outs,labels)
            loss.backward()
            optimizer.step()
            running+=loss.item()*imgs.size(0)
        train_loss=running/len(train_loader.dataset)
        train_losses.append(train_loss)
        # validation
        model.eval()
        vrunning=0.0
        with torch.no_grad():
            for imgs,labels in val_loader:
                imgs,labels=imgs.to(device),labels.to(device)
                outs=model(imgs)
                loss=criterion(outs,labels)
                vrunning+=loss.item()*imgs.size(0)
        val_loss=vrunning/len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f"[Epoch {epoch+1}/{epochs}] TrainLoss={train_loss:.6f} ValLoss={val_loss:.6f}")
        if val_loss<best_val:
            best_val=val_loss
            torch.save(model.state_dict(),save_path)
    print("Training finished. Best val loss:",best_val)
    # Plot losses
    plt.figure(figsize=(8,5))
    plt.plot(range(1,epochs+1),train_losses,label='Train Loss')
    plt.plot(range(1,epochs+1),val_losses,label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.title('Training & Validation Loss')
    plt.legend(); plt.grid(True)
    plt.savefig('training_loss_plot.png')
    print("Training loss plot saved as training_loss_plot.png")
    return model

# ---------------------------
# Predict AOI
# ---------------------------
def predict_aoi(model,img_path,transform,label_scale=90.0):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    img=Image.open(img_path).convert('RGB')
    x=transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out=model(x)
    return float(out[0][0]*label_scale)

# ---------------------------
# Overlay ellipse + predicted AOI
# ---------------------------
def overlay_aoi_on_image(img_path,pred_aoi_deg,save_path):
    img=cv2.imread(img_path)
    h,w,_=img.shape
    center=(w//2,h//2)
    a=w//4; b=max(1,int(a*math.sin(math.radians(pred_aoi_deg))))
    angle=0  # horizontal ellipse for visualization
    cv2.ellipse(img,center,(a,b),angle,0,360,(0,255,0),2)
    text=f"AOI={pred_aoi_deg:.1f}°"
    cv2.putText(img,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
    cv2.imwrite(save_path,img)

# ---------------------------
# Main
# ---------------------------
if __name__=="__main__":
    outdir='synthetic_dataset'
    n_images=500
    generate_synthetic_dataset(outdir,n_images)
    
    # split train/val
    df=pd.read_csv(os.path.join(outdir,'labels.csv'))
    train=df.sample(frac=0.8,random_state=42)
    val=df.drop(train.index)
    train.to_csv(os.path.join(outdir,'train_labels.csv'),index=False)
    val.to_csv(os.path.join(outdir,'val_labels.csv'),index=False)
    
    # train with visualization
    model=train_resnet(
        csv_train=os.path.join(outdir,'train_labels.csv'),
        csv_val=os.path.join(outdir,'val_labels.csv'),
        img_dir=outdir,
        epochs=20,
        batch_size=16,
        lr=0.0001,
        label_scale=90.0,
        save_path='resnet_aoi.pth'
    )
    
    # prediction transform
    transform_pred=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    
    # predict and overlay
    vis_dir='predicted_overlay'
    os.makedirs(vis_dir,exist_ok=True)
    for fname in os.listdir(outdir):
        if fname.endswith('.png'):
            img_path=os.path.join(outdir,fname)
            pred_aoi=predict_aoi(model,img_path,transform_pred,label_scale=90.0)
            overlay_aoi_on_image(img_path,pred_aoi,os.path.join(vis_dir,f'pred_{fname}'))
    print("Predicted AOI overlay images saved in:",vis_dir)
