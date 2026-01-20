from flask import Flask, render_template, request
import os
import torch
from torchvision import transforms
from fsp_aoi_full_pipeline import get_resnet_regression, predict_aoi, overlay_aoi_on_image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_RESULTS = 'static/results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_RESULTS, exist_ok=True)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_resnet_regression(pretrained=False)
model.load_state_dict(torch.load('resnet_aoi.pth', map_location=device))
model.to(device)
model.eval()

# Transform for prediction
transform_pred = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        # Save uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        print(f"[DEBUG] Uploaded file saved at: {filepath}")

        # Predict AOI
        pred_aoi = predict_aoi(model, filepath, transform_pred, label_scale=90.0)
        print(f"[DEBUG] Predicted AOI = {pred_aoi:.2f} degrees")

        # Save overlay result inside static/results
        result_filename = f'pred_{file.filename}'
        result_path = os.path.join(STATIC_RESULTS, result_filename)
        overlay_aoi_on_image(filepath, pred_aoi, result_path)
        print(f"[DEBUG] Overlay image saved at: {result_path}")

        # The browser will request this URL
        browser_url = f"/static/results/{result_filename}"
        print(f"[DEBUG] Browser should load: {browser_url}")

        return render_template(
            'result.html',
            pred_aoi=pred_aoi,
            img_file=f"results/{result_filename}"
        )

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
