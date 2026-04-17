import os

# Set your audio folder path
WAV_DIR = os.path.join("VoiceMos", "main", "DATA", "wav")

def check_audio_files(list_file_path):
    print(f"\n🔍 Scanning against: {list_file_path}")
    
    if not os.path.exists(list_file_path):
        print(f"❌ Error: Could not find {list_file_path} in this directory.")
        return

    # Read the text file
    with open(list_file_path, 'r') as file:
        lines = file.readlines()

    total_files = len(lines)
    found_files = []  # List to store the actual names and scores
    missing_count = 0

    for line in lines:
        # Split by comma to get filename and score
        parts = line.strip().split(',')
        wav_name = parts[0]
        
        # Grab the score so you don't have to search for it later
        human_score = parts[1] if len(parts) > 1 else "Unknown"
        
        full_path = os.path.join(WAV_DIR, wav_name)

        if os.path.exists(full_path):
            found_files.append((wav_name, human_score))
        else:
            missing_count += 1

    # Print the final report
    print(f"📊 Results:")
    print(f"   Audio Files Found:   {len(found_files)} / {total_files}")
    print(f"   Audio Files Missing: {missing_count} / {total_files}")

    # Safely print the first 5 found files with their human scores
    if len(found_files) > 0:
        print(f"   ✅ First 5 files ready for your Streamlit App:")
        for wav_name, score in found_files[:5]:
            print(f"      - File: {wav_name} | Actual Human MOS: {score}")

# Run the checker for both files
check_audio_files("VoiceMos/main/DATA/sets/val_mos_list.txt")
check_audio_files("VoiceMos/main/DATA/sets/test_mos_list.txt")