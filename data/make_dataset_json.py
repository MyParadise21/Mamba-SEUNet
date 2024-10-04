import argparse
import os
import json

def list_files_in_directory(directory_path):
    # List all files in the directory
    files = []
    for root, dirs, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.wav'):
                files.append(os.path.join(root, filename))
    return files

def save_files_to_json(files, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(files, json_file, indent=4)

def make_json(directory_path, output_file):
    # Get the list of files and save to JSON
    files = list_files_in_directory(directory_path)
    save_files_to_json(files, output_file)

# create training set json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="/Work21/2024/wangjunyu/pythonproject/Data_16k/")

    args = parser.parse_args()

    prepath = args.path if (args.path is not None) else "../"

    ## train_clean
    make_json(
        os.path.join(prepath, 'clean_train/'),
        'train_clean.json'
    )

    ## train_noisy
    make_json(
        os.path.join(prepath, 'noisy_train/'),
        'train_noisy.json'
    )

    ## valid_clean
    make_json(
         os.path.join(prepath, 'clean_valid/'),
        'valid_clean.json'
    )

    ## valid_noisy
    make_json(
        os.path.join(prepath, 'noisy_valid/'),
        'valid_noisy.json'
    )

    ## test_clean
    make_json(
       os.path.join(prepath, 'clean_test/'),
        'test_clean.json'
    )

    ## test_noisy
    make_json(
       os.path.join(prepath, 'noisy_test/'),
        'test_noisy.json'
    )
    # ----------------------------------------------------------#


if __name__ == '__main__':
    main()
