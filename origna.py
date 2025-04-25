import os
import re
import pandas as pd
import numpy as np
from sklearn import neighbors
import torch
import torch.nn as nn
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from torchvision.models import resnet18


# 1. Define a traditional CNN for feature extraction
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Adaptive pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x

class CNNWithTransformer(nn.Module):
    def __init__(self, feature_dim=256, transformer_layers=2):
        super().__init__()
        self.cnn = SimpleCNN()  # 用你的结构或改用 ResNet
        self.linear_proj = nn.Linear(256, 128)  # 将CNN特征转为Transformer输入维度
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        batch_size = x.size(0)
        features = self.cnn(x)  # shape: [B, 256]
        x = self.linear_proj(features).unsqueeze(1)  # [B, 1, 128]
        x = self.transformer(x)  # [B, 1, 128]
        x = x.squeeze(1)  # [B, 128]
        return x


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        resnet = resnet18(pretrained=True)
        # 去掉最后的全连接层
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # [B, 512, 1, 1]
        self.fc = nn.Linear(512, output_dim)  # 压缩为 Transformer 输入维度

    def forward(self, x):
        x = self.backbone(x)       # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten -> [B, 512]
        x = self.fc(x)             # -> [B, output_dim]
        return x


def normalize_coords(y):
    # 假设 x ∈ [0, 58], y ∈ [0, 84]
    y[:, 0] /= 58.0
    y[:, 1] /= 84.0
    return y

def denormalize_coords(y):
    y[:, 0] *= 58.0
    y[:, 1] *= 84.0
    return y

# 2. Function to extract features using our custom CNN
def extract_features(image_path, model, preprocess, device):
    try:
        image = Image.open(image_path).convert('RGB')
        image = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
        with torch.no_grad():
            features = model(image)
        return features.squeeze().cpu().numpy()  # Move to CPU and convert to numpy
    except Exception as e:
        print(f"无法处理图片 {image_path}: {e}")
        return None

# Recursive function to get all image paths
def natural_sort_key(path):
    folder_name = os.path.basename(os.path.dirname(path))  # e.g. '2', '12'
    file_name = os.path.basename(path)  # e.g. '0.jpg'
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('(\d+)', folder_name + file_name)]

def get_all_image_paths(folder):
    image_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    image_paths.sort(key=natural_sort_key)  # 按目录名 + 文件名进行自然排序
    return image_paths


# 3. Build graph structure (using coordinate KNN)
def build_adjacency_matrix(coordinates, n_neighbors=4):
    """
    Build adjacency matrix using KNN
    :param coordinates: Image coordinate matrix, shape (N, 2)
    :param n_neighbors: Number of neighbors per node
    :return: Adjacency matrix A
    """
    A = kneighbors_graph(coordinates, n_neighbors=n_neighbors, mode='connectivity', include_self=False)
    return A.toarray()  # Convert sparse matrix to dense

# 4. Define GCN model

class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True)
        self.conv3 = GATConv(hidden_channels * 4, hidden_channels, heads=1)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.residual = nn.Linear(in_channels, out_channels)  # 残差连接

    def forward(self, x, edge_index):
        residual = self.residual(x)
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = self.lin(x) + residual  # 添加残差
        return x


# 5. Train GCN model
def train_gcn(model, data, optimizer,epochs):
    """
    Train GCN model
    :param model: GCN model
    :param data: Graph data
    :param optimizer: Optimizer
    :param epochs: Training epochs
    :return: Training loss history
    """
    model.train()
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.huber_loss(out, data.y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    return loss_history

# 6. Predict coordinates for new images
def predict_coordinates(image_path, true_coords, model, preprocess, train_features, train_coordinates, cnn_model, device, n_neighbors=4):
    """
    Predict coordinates for new image and output true vs predicted coordinates
    :param image_path: Path to new image
    :param true_coords: True coordinates of new image
    :param model: Trained GCN model
    :param preprocess: Image preprocessing pipeline
    :param train_features: Training set feature matrix
    :param train_coordinates: Training set coordinate matrix
    :param cnn_model: CNN model
    :param device: Device (cuda/cpu)
    :param n_neighbors: Number of KNN neighbors
    :return: True and predicted coordinates (predicted coordinates are rounded and clamped)
    """
    # Extract features from new image
    features = extract_features(image_path, cnn_model, preprocess, device)
    if features is None:
        return true_coords, None

    # Add new image features and coordinates to training set
    test_feature = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(device)
    test_coord = torch.tensor(true_coords, dtype=torch.float).unsqueeze(0).to(device)

    combined_features = torch.cat([train_features, test_feature], dim=0)
    combined_coordinates = torch.cat([train_coordinates, test_coord], dim=0)

    # Build new graph structure
    A = build_adjacency_matrix(combined_coordinates.cpu().numpy(), n_neighbors=n_neighbors)
    edge_index = torch.tensor(np.array(np.where(A == 1)), dtype=torch.long).to(device)

    # Predict coordinates
    with torch.no_grad():
        predicted_coords = model(combined_features, edge_index)
    
    # Process predicted coordinates
    predicted_coords = predicted_coords[-1].cpu().numpy()  # Take last node (test image) predicted coordinates
    
    # 1. Round to nearest integer
    predicted_coords = np.round(predicted_coords).astype(int)
    
    # 2. Clamp x coordinate to [1, 58]
    predicted_coords[0] = np.clip(predicted_coords[0], 1, 58)
    
    # 3. Clamp y coordinate to [0, 84]
    predicted_coords[1] = np.clip(predicted_coords[1], 0, 84)
    
    return true_coords, predicted_coords

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize our custom CNN model
    # cnn_model = SimpleCNN().to(device)
    cnn_model = ResNetFeatureExtractor(output_dim=128).to(device)
    cnn_model.eval()  # Set to evaluation mode
    print("Constructing Model")

    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("Loading Data")

    train_excel_file = r'./train_positions.xlsx'
    train_df = pd.read_excel(train_excel_file, sheet_name='Sheet1')
    train_coordinates = train_df[['x', 'y']].values

    # 加载并排序训练图片路径
    train_image_folder = r'./train'
    train_image_paths = get_all_image_paths(train_image_folder)

    features_list = []
    for img_path in train_image_paths:
        features = extract_features(img_path, cnn_model, preprocess, device)
        if features is not None:
            features_list.append(features)
        else:
            print(f"Skipping invalid image: {img_path}")

    # Check feature dimensions
    print("Check feature dimensions")
    feature_dim = None
    for i, f in enumerate(features_list):
        if feature_dim is None:
            feature_dim = f.shape[0]
        elif f.shape[0] != feature_dim:
            print(f"Feature vector {i} has inconsistent dimension: {f.shape}")
            features_list[i] = np.zeros(feature_dim)  # Fill with zero vector

    features_array = np.array(features_list, dtype=np.float32)  # Convert to numpy array
    if features_array.shape[0] == 0:
        print("Error: No valid features extracted. Please check image paths and preprocessing.")
        return

    # 3. Read training coordinates from Excel file
    train_excel_file = r'./train_positions.xlsx'  # Replace with your training coordinates Excel file

    train_df = pd.read_excel(train_excel_file, sheet_name='Sheet1')  # Read data from Sheet1
    train_coordinates = train_df[['x', 'y']].values  # Extract x and y columns, convert to NumPy array

    # Check coordinate dimensions
    if train_coordinates.shape[0] != len(features_list):
        print("Error: Number of coordinates doesn't match number of features.")
        return

    # 4. Build graph structure
    n_neighbors = 4  # Number of KNN neighbors
    A = build_adjacency_matrix(train_coordinates, n_neighbors)
    edge_index = torch.tensor(np.array(np.where(A == 1)), dtype=torch.long).to(device)

    # 5. Prepare GCN data
    train_features = torch.tensor(features_array, dtype=torch.float).to(device)  # Node features
    train_coordinates = torch.tensor(train_coordinates, dtype=torch.float).to(device)  # Node coordinates (labels)
    data = Data(x=train_features, edge_index=edge_index, y=train_coordinates).to(device)

    # 6. Initialize GCN model
    print("Initialize GCN model")
    gcn_model = GATNet(features_array.shape[1], 128, 2).to(device)
    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=1e-4)

    # 7. Train GCN model and record loss
    print("Train GCN model")
    loss_history = train_gcn(gcn_model, data, optimizer,epochs=20000)

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.close()

    # 8. Read test images and their true coordinates
    # 读取测试坐标
    test_excel_file = r'./test_positions.xlsx'
    test_df = pd.read_excel(test_excel_file, sheet_name='Sheet1')
    test_coordinates = test_df[['x', 'y']].values

    # 加载并排序测试图片路径
    test_image_folder = r'./test-suiji4'
    test_image_paths = get_all_image_paths(test_image_folder)  # 按文件名排序

    print(len(test_image_paths))
    print(len(test_coordinates))

    print()
    # 9. Batch prediction and collect results
    results = []
    distances = []
    for image_path, true_coords in zip(test_image_paths, test_coordinates):
        true_coords, predicted_coords = predict_coordinates(
            image_path, true_coords, gcn_model, preprocess,
            train_features, train_coordinates, cnn_model, device, n_neighbors
        )
        if predicted_coords is not None:
            # Calculate Euclidean distance using integer coordinates
            distance = euclidean(true_coords, predicted_coords)
            distances.append(distance)

            # Add to results list
            results.append({
                'image_name': os.path.basename(image_path),
                'true_x': true_coords[0],
                'true_y': true_coords[1],
                'pred_x': predicted_coords[0],
                'pred_y': predicted_coords[1],
                'distance': distance
            })

            print(f"Image: {os.path.basename(image_path)}")
            print("True coordinates:", true_coords)
            print("Predicted coordinates (rounded and clamped):", predicted_coords)
            print("Euclidean distance:", distance)
            print()

    # 10. Save results to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel('prediction_results.xlsx', index=False)

    # 11. Calculate error statistics
    if distances:
        max_error = max(distances)
        min_error = min(distances)
        avg_error = sum(distances) / len(distances)

        print(f"\nError Statistics:")
        print(f"Max error: {max_error:.2f}")
        print(f"Min error: {min_error:.2f}")
        print(f"Average error: {avg_error:.2f}")

        # 12. Plot true vs predicted coordinates
        plt.figure(figsize=(10, 10))
        plt.scatter(results_df['true_x'], results_df['true_y'], c='blue', label='True Positions')
        plt.scatter(results_df['pred_x'], results_df['pred_y'], c='red', label='Predicted Positions')

        # Draw connecting lines
        for _, row in results_df.iterrows():
            plt.plot([row['true_x'], row['pred_x']], [row['true_y'], row['pred_y']], 'k--', alpha=0.3)

        plt.title('True vs Predicted Positions')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid()
        plt.savefig('positions_comparison.png')
        plt.close()

        # 13. Plot cumulative error distribution
        plt.figure(figsize=(10, 5))
        sorted_distances = np.sort(distances)
        cdf = np.arange(1, len(sorted_distances)+1) / len(sorted_distances)
        plt.plot(sorted_distances, cdf)
        plt.title('Cumulative Distribution of Errors')
        plt.xlabel('Euclidean Distance Error')
        plt.ylabel('Cumulative Probability')
        plt.grid()
        plt.savefig('error_distribution.png')
        plt.close()
    else:
        print("No valid prediction results to analyze")

if __name__ == '__main__':
    main()