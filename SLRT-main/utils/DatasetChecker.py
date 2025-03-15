import os
import json


def verifica_video(cartella, json_filename, all_videos):
    json_path = os.path.join(cartella, json_filename)
    videos_path = os.path.join(cartella, "Videos")

    if not os.path.exists(json_path):
        print(f"❌ File JSON non trovato: {json_path}")
        return
    if not os.path.exists(videos_path):
        print(f"❌ Cartella Video non trovata: {videos_path}")
        return

    # Carica il file JSON
    with open(json_path, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            print(f"❌ Errore nel parsing di {json_path}")
            return

    # Estrai gli ID e verifica la presenza dei video
    id_video_json = {str(item["id"]) + ".mp4" for item in data}  # Set di ID + .mp4
    video_nella_cartella = {nome for nome in os.listdir(videos_path) if
                            nome.endswith(".mp4")}  # Set di file mp4 nella cartella

    # Trova i mancanti
    mancanti = id_video_json - video_nella_cartella

    if mancanti:
        print(f"⚠️ Video mancanti in {videos_path}:")
        for video in mancanti:
            print(f"  - {video}")
    else:
        print(f"✅ Tutti i video sono presenti in {videos_path}!")

    # Controllo duplicati in altre cartelle
    duplicati = video_nella_cartella & all_videos
    if duplicati:
        print(f"❌ Video duplicati trovati in altre cartelle per {cartella}:")
        for video in duplicati:
            print(f"  - {video}")

    # Aggiungi i video attuali all'insieme globale
    all_videos.update(video_nella_cartella)


if __name__ == "__main__":
    dataset_path = "D:/Computer Science - AI/Primo anno/Secondo semestre/Computer vision/Dataset prova"
    json_files = {"Train": "train.json", "Test": "test.json", "Val": "val.json"}

    all_videos = set()  # Insieme per tenere traccia di tutti i video

    for folder, json_filename in json_files.items():
        print(f"🔍 Controllo per {folder}...")
        verifica_video(os.path.join(dataset_path, folder), json_filename, all_videos)
        print("-")
