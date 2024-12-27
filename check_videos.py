import json
from pathlib import Path

def check_videos():
    json_path = Path('data/WLASL_v0.3.json')
    video_dir = Path('data/videos')
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get all video IDs from JSON
    video_ids = set()
    for entry in data:
        if entry['instances']:
            for instance in entry['instances']:
                video_ids.add(f"{instance['video_id']}.mp4")
    
    # Check actual videos
    actual_videos = set(f.name for f in video_dir.glob('*.mp4'))
    
    print(f"Total videos in JSON: {len(video_ids)}")
    print(f"Total videos in directory: {len(actual_videos)}")
    print(f"Missing videos: {len(video_ids - actual_videos)}")
    
    if len(video_ids - actual_videos) > 0:
        print("\nFirst 5 missing videos:")
        for video in sorted(video_ids - actual_videos)[:5]:
            print(video)

if __name__ == '__main__':
    check_videos() 