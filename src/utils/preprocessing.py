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

def get_subjects(base_path):
    subjects = set()

    for folder in ["Open_Eyes", "Closed_Eyes"]:
        folder_path = os.path.join(base_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                subject_id = file.split('_')[0]
                subjects.add(subject_id)

    return list(subjects)


def split_subjects(subjects, test_ratio=0.3):
    random.shuffle(subjects)
    split_index = int(len(subjects) * test_ratio)
    return subjects[split_index:], subjects[:split_index]  # train, test


def copy_files(base_path, train_subjects, test_subjects):
    for category in ["Open_Eyes", "Closed_Eyes"]:
        source_dir = os.path.join(base_path, category)
        train_dir = os.path.join("data/train", category)
        test_dir = os.path.join("data/test", category)

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        train_count = 0
        test_count = 0

        for file in os.listdir(source_dir):
            if not file.endswith(('.png', '.jpg', '.jpeg')):
                continue

            subject_id = file.split('_')[0]
            src = os.path.join(source_dir, file)

            if subject_id in train_subjects:
                shutil.copy(src, os.path.join(train_dir, file))
                train_count += 1
            elif subject_id in test_subjects:
                shutil.copy(src, os.path.join(test_dir, file))
                test_count += 1

        print(f"{category}")
        print(f"Train: {train_count} | Test: {test_count}\n")

def balance_train_data(train_dir):
    open_dir = os.path.join(train_dir, "Open_Eyes")
    closed_dir = os.path.join(train_dir, "Closed_Eyes")

    open_files = os.listdir(open_dir)
    closed_files = os.listdir(closed_dir)

    min_count = min(len(open_files), len(closed_files))

    # Randomly select equal number
    open_sample = random.sample(open_files, min_count)
    closed_sample = random.sample(closed_files, min_count)

    # Remove extra files
    for file in open_files:
        if file not in open_sample:
            os.remove(os.path.join(open_dir, file))

    for file in closed_files:
        if file not in closed_sample:
            os.remove(os.path.join(closed_dir, file))

    print(f"Balanced training set to {min_count} per class\n")
if __name__ == "__main__":
    base_path = "data/raw"

    # Step 1: Analyze dataset
    run_glasses_analysis(base_path)

    # Step 2: Get all subjects
    subjects = get_subjects(base_path)

    # Step 3: Split subjects
    train_subjects, test_subjects = split_subjects(subjects)

    print(f"Train Subjects: {len(train_subjects)}")
    print(f"Test Subjects: {len(test_subjects)}\n")

    # Step 4: Copy files
    copy_files(base_path, train_subjects, test_subjects)

    balance_train_data("data/train")

    print(f"Train Subjects: {len(train_subjects)}")
    print(f"Test Subjects: {len(test_subjects)}\n")
