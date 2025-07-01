import os

def search_for_video():
    found_matches = []
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk("."):
        for filename in files:
            filepath = os.path.join(root, filename)
            
            try:
                # Try to open each file as text
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    
                    # Search for "video" (case-insensitive)
                    if "video" in content.lower():
                        # Count occurrences
                        count = content.lower().count("video")
                        found_matches.append((filepath, count))
                        print(f"Found 'video' {count} time(s) in: {filepath}")
                        
            except (UnicodeDecodeError, PermissionError, OSError, IsADirectoryError) as e:
                # Skip files that can't be read as text or have permission issues
                print(f"Skipped {filepath}: {type(e).__name__}")
                continue
    
    if not found_matches:
        print("No files containing 'video' found in current directory tree.")
    else:
        print(f"\nSummary: Found 'video' in {len(found_matches)} file(s)")
        
        # Optional: Show total occurrences
        total_occurrences = sum(count for _, count in found_matches)
        print(f"Total occurrences: {total_occurrences}")

if __name__ == "__main__":
    search_for_video()