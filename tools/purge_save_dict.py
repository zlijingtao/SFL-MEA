import os

def delete_files_with_name(folder_path, file_name):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file_name in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

# Example usage
folder_path = "./saves"
specific_name = "_50.tar"
delete_files_with_name(folder_path, specific_name)

specific_name = "_150.tar"
delete_files_with_name(folder_path, specific_name)

specific_name = "_100.tar"
delete_files_with_name(folder_path, specific_name)