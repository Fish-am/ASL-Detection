import json
from pathlib import Path

def check_json():
    json_path = Path('data/WLASL_v0.3.json')
    
    print(f"Checking JSON file at {json_path.absolute()}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"\nJSON structure:")
    print(f"Type of data: {type(data)}")
    print(f"Number of entries: {len(data)}")
    
    if len(data) > 0:
        print("\nFirst entry structure:")
        first_entry = data[0]
        for key, value in first_entry.items():
            print(f"{key}: {type(value)}")
            
        if 'instances' in first_entry and first_entry['instances']:
            print("\nFirst instance structure:")
            first_instance = first_entry['instances'][0]
            for key, value in first_instance.items():
                print(f"{key}: {type(value)}")

if __name__ == '__main__':
    check_json() 