import os
import random
import shutil

# Lagay lang ito para same ouput each run
random.seed(67)

def check_glasses(folder_path, state_label):
    glasses_count = 0
    no_glasses_count = 0

    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return 0, 0

    for filename in os.listdir(folder_path):
        parts = filename.split('_')

        # Ensure filename format is valid
        if len(parts) >= 4:
            glasses_val = parts[3]

            if glasses_val == '1':
                glasses_count += 1
            else:
                no_glasses_count += 1

    print(f"--- {state_label} ---")
    print(f"With Glasses: {glasses_count}")
    print(f"No Glasses: {no_glasses_count}\n")

    return glasses_count, no_glasses_count


def run_glasses_analysis(base_path):
    open_eyes_dir = os.path.join(base_path, "Open_Eyes")
    closed_eyes_dir = os.path.join(base_path, "Closed_Eyes")

    open_g, open_no = check_glasses(open_eyes_dir, "OPEN EYES")
    closed_g, closed_no = check_glasses(closed_eyes_dir, "CLOSED EYES")

    total = open_g + open_no + closed_g + closed_no

    print(f"Total Images Processed: {total}")

    return {
        "open_with_glasses": open_g,
        "open_without_glasses": open_no,
        "closed_with_glasses": closed_g,
        "closed_without_glasses": closed_no,
        "total": total
    }


def split_data(source_dir, train_dir, test_dir, test_ratio=0.15):
    # Create folders if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    files = os.listdir(source_dir)
    random.shuffle(files)

    split_index = int(len(files) * test_ratio)

    test_files = files[:split_index]
    train_files = files[split_index:]

    # Copy files
    for file in train_files:
        shutil.copy(
            os.path.join(source_dir, file),
            os.path.join(train_dir, file)
        )

    for file in test_files:
        shutil.copy(
            os.path.join(source_dir, file),
            os.path.join(test_dir, file)
        )

    print(f"{source_dir}")
    print(f"Train: {len(train_files)} | Test: {len(test_files)}\n")



if __name__ == "__main__":
    base_path = "data/raw"

    # Step 1: Analyze dataset
    run_glasses_analysis(base_path)

    # Step 2: Split dataset
    split_data(
        os.path.join(base_path, "Open_Eyes"),
        "data/train/Open_Eyes",
        "data/test/Open_Eyes"
    )

    split_data(
        os.path.join(base_path, "Closed_Eyes"),
        "data/train/Closed_Eyes",
        "data/test/Closed_Eyes"
    )