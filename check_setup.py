from pathlib import Path

def check_setup():
    required_files = {
        'data/WLASL_v0.3.json': 'JSON file',
        'data/videos': 'Videos directory'
    }
    
    for path, description in required_files.items():
        p = Path(path)
        if not p.exists():
            print(f"ERROR: {description} not found at {p.absolute()}")
        else:
            print(f"✓ Found {description}")

if __name__ == '__main__':
    check_setup()