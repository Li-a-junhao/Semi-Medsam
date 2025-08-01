import os
import json

DATASET_DIR = 'demo_dataset'
OUTPUT_JSON = 'annotation_demo.json'


def create_annotation_file_flat():
    annotations = []
    print(f"[*] Starting to scan directory: {DATASET_DIR}")
    all_files = sorted(os.listdir(DATASET_DIR))

    for filename in all_files:
        if filename.endswith('_1.png'):
            base_name = filename.replace('_1.png', '')

            wli_label_filename = f"{base_name}_1_label.png"
            nbi_image_filename = f"{base_name}_2.png"

            if wli_label_filename in all_files and nbi_image_filename in all_files:
                wli_image_path = os.path.abspath(os.path.join(DATASET_DIR, filename))
                wli_label_path = os.path.abspath(os.path.join(DATASET_DIR, wli_label_filename))
                nbi_image_path = os.path.abspath(os.path.join(DATASET_DIR, nbi_image_filename))

                data_entry = {
                    "id": filename,
                    "image_path": wli_image_path,
                    "label_path": wli_label_path,
                    "mode_nbi_image_path": nbi_image_path,
                }

                annotations.append(data_entry)
            else:
                print(f"[!] Warning: Skipping '{filename}' because its corresponding label or NBI file is missing.")

    final_data_structure = {
        "demo": annotations
    }
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_data_structure, f, indent=4)
    print(f"\n[*] Successfully generated '{OUTPUT_JSON}' with {len(annotations)} entries.")


if __name__ == '__main__':
    create_annotation_file_flat()
