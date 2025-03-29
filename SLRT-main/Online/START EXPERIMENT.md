1. git clone https://github.com/Federico-uni/CV-Project.git
2. cd CV-Project
3. python -m venv venv (3.10)
4. venv\Scripts\activate | source venv/bin/activate
5. pip install -r SLRT-main/Online/training_requirements.txt
6. git clone --recursive https://github.com/parlance/ctcdecode.gitcd ctcdecode && pip install .
7. Link da dove scaricare il dataset per usarlo direttamente: https://drive.google.com/file/d/1GvPwVPmjQQgEAucEVhlPmkZs9YAfXv3o/view?usp=drive_link
8. cambiare il percorso del dataset su dataset_noKeypoints.yaml (CV-Project/SLRT-main/Online/CSLR/configs/dataset_noKeypoints.yaml) --riga 12
9. cambiare il percorso della cartella dei risultati su dataset_noKeypoints.yaml (CV-Project/SLRT-main/Online/CSLR/configs/dataset_noKeypoints.yaml) --riga 37
10. torchrun --nproc_per_node=1 SLRT-main/Online/CSLR/training.py --config CV-Project/SLRT-main/Online/CSLR/configs/dataset_noKeypoints.yaml
11. OPZONALE link dove scaricare i pretrained model (kaggle):
    - https://www.kaggle.com/datasets/beascavo/pretrained-models-andkp2-0
    - https://www.kaggle.com/datasets/beascavo/hrnet-models