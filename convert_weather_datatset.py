import json
import os
from sklearn.model_selection import train_test_split

def convert_weather_to_stargan_format():
    # Load your dataset
    with open('data/weather/dataset/dataset.json', 'r') as f:
        data = json.load(f)
    
    # Create attribute mapping (binary format like CelebA)
    weather_attrs = ['clear', 'rain', 'fog', 'snow']
    
    # Create train/test split
    train_data, test_data = train_test_split(data['labels'], test_size=0.2, random_state=42)
    
    # Create attribute files
    create_attr_file(train_data, 'data/weather/list_attr_weather_train.txt', weather_attrs)
    create_attr_file(test_data, 'data/weather/list_attr_weather_test.txt', weather_attrs)
    
    print(f"Created training set: {len(train_data)} images")
    print(f"Created test set: {len(test_data)} images")

def create_attr_file(data, output_path, weather_attrs):
    with open(output_path, 'w') as f:
        # Write header
        f.write(f"{len(data)}\n")
        f.write(" ".join(weather_attrs) + "\n")
        
        # Write data
        for img_path, label in data:
            # Create binary encoding (one-hot)
            attrs = [-1] * len(weather_attrs)
            attrs[label] = 1
            
            # Format: filename attr1 attr2 attr3 attr4
            filename = img_path.replace('/', '_')  # Replace / with _
            attr_str = " ".join(map(str, attrs))
            f.write(f"{filename} {attr_str}\n")

def reorganize_images():
    """Reorganize images to flat structure for StarGAN"""
    import shutil
    
    # Create directories
    train_dir = 'data/weather/train'
    test_dir = 'data/weather/test'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Load dataset
    with open('data/weather/dataset/dataset.json', 'r') as f:
        data = json.load(f)
    
    # Split data
    train_data, test_data = train_test_split(data['labels'], test_size=0.2, random_state=42)
    
    # Copy training images
    for img_path, label in train_data:
        src = f'data/weather/dataset/{img_path}'
        dst = f'{train_dir}/{img_path.replace("/", "_")}'
        if os.path.exists(src):
            shutil.copy2(src, dst)
    
    # Copy test images
    for img_path, label in test_data:
        src = f'data/weather/dataset/{img_path}'
        dst = f'{test_dir}/{img_path.replace("/", "_")}'
        if os.path.exists(src):
            shutil.copy2(src, dst)

if __name__ == "__main__":
    convert_weather_to_stargan_format()
    reorganize_images()