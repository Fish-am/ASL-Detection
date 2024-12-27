import json
from pathlib import Path

def check_missing_videos():
    json_path = Path('data/WLASL_v0.3.json')
    video_dir = Path('data/videos')
    
    # Get available videos
    available_videos = set(f.stem for f in video_dir.glob('*.mp4'))
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get required videos
    required_videos = set()
    for item in data:
        if item['instances']:
            for instance in item['instances']:
                required_videos.add(instance['video_id'])
    
    # Calculate missing videos
    missing_videos = required_videos - available_videos
    
    print(f"Total required videos: {len(required_videos)}")
    print(f"Available videos: {len(available_videos)}")
    print(f"Missing videos: {len(missing_videos)}")
    
    if missing_videos:
        print("\nFirst 10 missing videos:")
        for video_id in sorted(missing_videos)[:10]:
            print(video_id)

if __name__ == '__main__':
    check_missing_videos() 