import json
from pathlib import Path

def verify_dataset():
    json_path = Path("data/WLASL_v0.3.json")
    video_dir = Path("data/videos")
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)['root']
    
    # Count expected videos
    expected_videos = set()
    for entry in data:
        if entry['instances'] is not None:
            for instance in entry['instances']:
                expected_videos.add(f"{instance['video_id']}.mp4")
    
    # Check actual videos
    actual_videos = set(video_dir.glob("*.mp4"))
    actual_video_names = {v.name for v in actual_videos}
    
    # Report
    print(f"Expected videos: {len(expected_videos)}")
    print(f"Downloaded videos: {len(actual_videos)}")
    print(f"Missing videos: {len(expected_videos - actual_video_names)}")
    
    if len(expected_videos - actual_video_names) > 0:
        print("\nMissing video IDs:")
        for video in sorted(expected_videos - actual_video_names)[:10]:
            print(video)
        if len(expected_videos - actual_video_names) > 10:
            print("... and more")

if __name__ == "__main__":
    verify_dataset() 