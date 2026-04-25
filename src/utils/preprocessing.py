import os

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


# Optional: run directly
if __name__ == "__main__":
    base_path = "data/train"

    run_glasses_analysis(base_path)