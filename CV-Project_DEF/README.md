# LIS

## Clone the Repository

To download this project from GitHub, run:

```bash
git clone https://github.com/Federico-uni/CV-Project.git
cd CV-Project/CV-Project_DEF
```

## Environment Setup

To set up the virtual environment with all required dependencies:

1. **Create and activate the environment using `conda`:**
   ```bash
   conda env create -f environment.yaml
   conda activate <env_name>  # Replace <env_name> with the actual environment name in the YAML
   ```
2. **Export intern libraries**
   ```bash
   export LD_PRELOAD=/ext/home/scavalent/miniconda3/envs/cv_env/x86_64-conda-linux-gnu/lib/libstdc++.so.6
   ```

3. **Build `ctcdecode`:**
   ```bash
   bash rebuild_ctcdecode.sh
   ```

4. **Install FFmpeg and Lintel:**
   ```bash
   bash install_ffmpeg_and_lintel.sh
   ```

5. **Build and install the custom C extension:**
   ```bash
   python setup.py install
   ```

###  Note

The scripts and files mentioned above can be found in the `<Virtual Environment Setup>` directory within this project.
---

## Dataset and Model Links

Please download the required datasets and pretrained models from the following links:

- **Dataset:**
  - [Download Continuous Data](https://unibari-my.sharepoint.com/:u:/g/personal/f_valentino7_studenti_uniba_it/EVF35QQHTTtJpklrk30VJrMBFD8dWWOBHAtN8UpubLrimw?e=OfMxAT)

- **Pretrained Models:**
  - [Isolated_training](https://unibari-my.sharepoint.com/:u:/g/personal/b_scavo_studenti_uniba_it/EWXvFzF5VltDo_JrtOhk4hIBPPczupz8huf2ItI6odeRyg?e=yLMnZG)
  - [MBart_pt](https://unibari-my.sharepoint.com/:u:/g/personal/b_scavo_studenti_uniba_it/EVJGswSwIMdCt8vxXVhOImkB78IVXrwBXQhJMynYJTZfew?e=EYo5Xc)

---

## Training Instructions

To launch the training process, follow these steps:

1. **Activate the environment:**
   ```bash
   conda activate <env_name>
   ```

2. **Run the training script:**
   Training on Isolated Videos
   ```bash
   torchrun --nproc_per_node=1 training.py --config IsolatedLIS/ISLIS_training_noKeypoints.yaml
   ```
   Training on Continuous Videos
   ```bash
   torchrun --nproc_per_node=1 training.py --config ContinuousLIS/CSLR_slide_noKpoints.yaml
   ```


> Make sure to adapt the paths and arguments based on your actual script layout and options.

---
