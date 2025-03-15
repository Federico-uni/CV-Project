import shutil
import json
import os

# Percorsi
dataset_dir = "D:/Computer Science - AI/Primo anno/Secondo semestre/Computer vision/Dataset prova"
isolated_dir = "D:/Computer Science - AI/Primo anno/Secondo semestre/Computer vision/Isolated/Isolated"

train_dir = os.path.join(dataset_dir, "Train")
test_dir = os.path.join(dataset_dir, "Test")
val_dir = os.path.join(dataset_dir, "Val")

json_file_path = os.path.join(isolated_dir, "isolatedLIS.json")
videos_dir_path = os.path.join(isolated_dir, "raw_videos")

# Carica i JSON esistenti
def carica_json(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

train_json = carica_json(os.path.join(train_dir, "train.json"))
test_json = carica_json(os.path.join(test_dir, "test.json"))
val_json = carica_json(os.path.join(val_dir, "val.json"))

# Assicura che le cartelle esistano
for path in [os.path.join(train_dir, "Videos"), os.path.join(test_dir, "Videos"), os.path.join(val_dir, "Videos")]:
    os.makedirs(path, exist_ok=True)

# Set con solo file .mp4
file_set = {nome for nome in os.listdir(videos_dir_path) if nome.endswith(".mp4")}
# Suddivisione dataset
train_size = int(len(file_set) * 0.7)
val_size = int(len(file_set) * 0.1)
test_size = len(file_set) - (train_size + val_size)

# Legge il file JSON principale
with open(json_file_path, "r", encoding="utf-8") as json_file:
    json_data = json.load(json_file)


# Iterazione per assegnare i video
i = 0
set_considered_videos = set()
for video_data in json_data:
    video_name = video_data["id"] + ".mp4"
    if video_name in file_set and (not(video_name in set_considered_videos)):
        if i < train_size:
            shutil.copy2(os.path.join(videos_dir_path, video_name), os.path.join(train_dir, "Videos", video_name))
            train_json.append(video_data)
            set_considered_videos.add(video_name)
        elif i>=train_size and i < train_size + val_size:
            shutil.copy2(os.path.join(videos_dir_path, video_name), os.path.join(val_dir, "Videos", video_name))
            val_json.append(video_data)
            set_considered_videos.add(video_name)
        else:
            shutil.copy2(os.path.join(videos_dir_path, video_name), os.path.join(test_dir, "Videos", video_name))
            test_json.append(video_data)
            set_considered_videos.add(video_name)
        i += 1
print(i)
print(len(file_set))
# Salvataggio JSON aggiornati
with open(os.path.join(train_dir, "train.json"), "w", encoding="utf-8") as json_train_file:
    json.dump(train_json, json_train_file, indent=4, ensure_ascii=False)

with open(os.path.join(test_dir, "test.json"), "w", encoding="utf-8") as json_test_file:
    json.dump(test_json, json_test_file, indent=4, ensure_ascii=False)

with open(os.path.join(val_dir, "val.json"), "w", encoding="utf-8") as json_val_file:
    json.dump(val_json, json_val_file, indent=4, ensure_ascii=False)

print("✅ Done")
