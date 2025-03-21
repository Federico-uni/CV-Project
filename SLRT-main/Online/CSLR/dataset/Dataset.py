import torch, pickle
import json, os, gzip
from glob import glob
import numpy as np
from utils.misc import get_logger
from collections import defaultdict


Hrnet_Part2index = {
    'pose': list(range(11)),
    'hand': list(range(91, 133)),
    'mouth': list(range(71,91)),
    'face_others': list(range(23, 71))
}
for k_ in ['mouth','face_others', 'hand']:
    Hrnet_Part2index[k_+'_half'] = Hrnet_Part2index[k_][::2]
    Hrnet_Part2index[k_+'_1_3'] = Hrnet_Part2index[k_][::3]
    
def get_keypoints_num(keypoint_file, use_keypoints):
    keypoints_num = 0
    assert 'hrnet' in keypoint_file
    Part2index = Hrnet_Part2index
    for k in sorted(use_keypoints):
        keypoints_num += len(Part2index[k])     
    return keypoints_num


class ISLRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_cfg, split, task='ISLR'):
        super(ISLRDataset, self).__init__()
        self.split = split #train, dev, test
        self.dataset_cfg = dataset_cfg
        self.root = os.path.join(*(self.dataset_cfg[split].split('/')[:-1]))
        self.vfile2raw_vlens = {}
        if 'MSASL' in dataset_cfg['dataset_name']:
            self.vocab = self.create_vocab()
            self.annotation = self.load_annotations(split)
        else:
            self.annotation = self.load_annotations(split)
            self.vocab = self.create_vocab()
        # print(len(self.vocab))
        self.input_streams = dataset_cfg.get('input_streams', ['rgb'])
        self.logger = get_logger()
        self.name2keypoints = self.load_keypoints()
        self.word_emb_tab = None
        if dataset_cfg.get('word_emb_file', None):
            self.word_emb_tab = self.load_word_emb_tab()
        self.other_vars = self.load_other_variants()

        self.g2g_input = None
        if task in ['G2G', 'bag_denoise']:
            with open(dataset_cfg['{}_extra_input'.format(split)], 'rb') as f:
                self.g2g_input = pickle.load(f)
        self.vfile2framelabel = self.create_framelabel()

    def load_keypoints(self):
        if 'keypoint' in self.input_streams or 'keypoint_coord' in self.input_streams or 'trajectory' in self.input_streams:
            with open(self.dataset_cfg['keypoint_file'], 'rb') as f:
                name2all_keypoints = pickle.load(f)
    
            assert 'hrnet' in self.dataset_cfg['keypoint_file']
            self.logger.info('Keypoints source: hrnet')
            Part2index = Hrnet_Part2index
    
            name2keypoints = {}
            for name, all_keypoints in name2all_keypoints.items():
                all_keypoints = np.array(all_keypoints)  # 👈 Fix importante: cast a NumPy array
                name2keypoints[name] = []
    
                for k in sorted(self.dataset_cfg['use_keypoints']):
                    selected_index = Part2index[k]
                    name2keypoints[name].append(all_keypoints[:, selected_index])  # T, N, 3
    
                name2keypoints[name] = np.concatenate(name2keypoints[name], axis=1)  # T, N, 3
                self.keypoints_num = name2keypoints[name].shape[1]
    
            self.logger.info(f'Total #={self.keypoints_num}') 
            assert self.keypoints_num == get_keypoints_num(
                self.dataset_cfg['keypoint_file'], 
                self.dataset_cfg['use_keypoints']
            )
    
        else:
            name2keypoints = None
    
        return name2keypoints
        
    def load_annotations(self, split):
        print("\n========== DEBUG: load_annotations() ==========")
        print(f"Split richiesto: {split}")
        print(f"Dataset Config: {self.dataset_cfg}")
        
        # Controlla se lo split esiste nel dataset_cfg
        if split not in self.dataset_cfg:
            raise ValueError(f"Errore: lo split '{split}' non è presente in dataset_cfg! Controlla il file YAML.")
        
        self.annotation_file = self.dataset_cfg[split]
        print(f"Percorso annotation_file (da YAML): {self.annotation_file}")
        
        # Convertire il percorso in assoluto se necessario
        if not os.path.isabs(self.annotation_file):
            self.annotation_file = os.path.abspath(self.annotation_file)
        print(f"Percorso assoluto annotation_file: {self.annotation_file}")
        
        # Verifica se il file esiste
        if not os.path.exists(self.annotation_file):
            raise FileNotFoundError(f"Errore: il file '{self.annotation_file}' non esiste! Controlla il percorso nel file YAML.")
        
        # Determina la root della directory
        self.root = os.path.dirname(self.annotation_file)
        print(f"Root directory: {self.root}")
        
        # Caricamento dell'annotazione con gestione degli errori
        try:
            with open(self.annotation_file, 'rb') as f:
                annotation = pickle.load(f)
            print("File caricato correttamente con pickle.")
        except Exception as e:
            print(f"Errore con pickle: {e}, tentando con gzip...")
            try:
                with gzip.open(self.annotation_file, 'rb') as f:
                    annotation = pickle.load(f)
                print("File caricato correttamente con gzip.")
            except Exception as e:
                raise RuntimeError(f"Errore nel caricamento del file '{self.annotation_file}': {e}")
        
        # Se annotation è un dizionario, convertilo in una lista
        if isinstance(annotation, dict):
            annotation = list(annotation.values())
        
        # Pulizia per dataset specifici
        if 'WLASL' in self.dataset_cfg['dataset_name']:
            variant_file = os.path.join(self.root, self.dataset_cfg['dataset_name'].split('_')[-1] + '.json')
            print(f"Caricamento variant file: {variant_file}")
            
            if not os.path.exists(variant_file):
                raise FileNotFoundError(f"Errore: il file '{variant_file}' non esiste!")
            
            with open(variant_file, 'r') as f:
                variant = json.load(f)
            
            annotation = [item for item in annotation if 'augmentation' not in item['video_file'] and item['name'] in variant]
            print(f"Annotazioni filtrate per WLASL: {len(annotation)}")
        
        elif 'MSASL' in self.dataset_cfg['dataset_name']:
            annotation = [item for item in annotation if item['label'] in self.vocab]
            print(f"Annotazioni filtrate per MSASL: {len(annotation)}")
        
        print("================================================\n")
        return annotation

    
    def load_word_emb_tab(self):
        fname = self.dataset_cfg['word_emb_file']
        with open(fname, 'rb') as f:
            word_emb_tab = pickle.load(f)
        return word_emb_tab
    
    def create_vocab(self):
        print("DEBUG: Avvio di create_vocab. Dataset:", self.dataset_cfg['dataset_name'])
        
        if 'WLASL' in self.dataset_cfg['dataset_name'] or 'NMFs-CSL' in self.dataset_cfg['dataset_name']:
            print("DEBUG: Branch WLASL/NMFs-CSL selezionato.")
            annotation = self.load_annotations('train')
            print("DEBUG: Annotation caricata, numero totale:", len(annotation))
            vocab = []
            for item in annotation:
                if item['label'] not in vocab:
                    vocab.append(item['label'])
                    print("DEBUG: Aggiunto label:", item['label'])
            vocab = sorted(vocab)
            print("DEBUG: Vocabolario ordinato:", vocab)
        
        elif 'MSASL' in self.dataset_cfg['dataset_name']:
            print("DEBUG: Branch MSASL selezionato.")
            msasl_file = os.path.join(self.root, 'MSASL_classes.json')
            print("DEBUG: Apertura file:", msasl_file)
            with open(msasl_file, 'rb') as f:
                all_vocab = json.load(f)
            print("DEBUG: Caricato all_vocab con", len(all_vocab), "voci")
            num = int(self.dataset_cfg['dataset_name'].split('_')[-1])
            print("DEBUG: Utilizzo delle prime", num, "voci di all_vocab")
            vocab = all_vocab[:num]
        
        elif self.dataset_cfg['dataset_name'] in ['phoenix_iso', 'phoenix2014_iso', 'phoenix_comb_iso', 'phoenix', 'phoenix2014', 'phoenixcomb', 'csl', 'csl_iso', 'IsolatedLIS']:
            print("DEBUG: Branch phoenix/csl selezionato.")
            vocab_file = self.dataset_cfg['vocab_file']
            print("DEBUG: Apertura file vocab:", vocab_file)
            with open(vocab_file, 'rb') as f:
                vocab = json.load(f)
            print("DEBUG: Vocabolario caricato:", vocab)
            if '<blank>' in vocab:
                if vocab.index('<blank>') != 0:
                    print("DEBUG: Attenzione: '<blank>' non è all'indice 0!")
                else:
                    print("DEBUG: '<blank>' è correttamente all'indice 0.")
                assert vocab.index('<blank>') == 0
            
            if 'iso' in self.dataset_cfg['dataset_name']:
                print("DEBUG: Variante 'iso' rilevata nel nome del dataset.")
                if self.dataset_cfg['dataset_name'] == 'phoenix_iso':
                    file_path = '../../data/phoenix_2014t/phoenix14t.{}'.format(self.split)
                    print("DEBUG: Apertura file gzip:", file_path)
                    with gzip.open(file_path, 'rb') as f:
                        ori_meta = pickle.load(f)
                    print("DEBUG: Metadati originali caricati, numero totale:", len(ori_meta))
                    for item in ori_meta:
                        self.vfile2raw_vlens[item['name']] = item['num_frames']
                        print("DEBUG: Impostato vfile2raw_vlens per", item['name'], "a", item['num_frames'])
        else:
            print("DEBUG: Nessuna condizione corrisponde al dataset specificato. Vocabolario vuoto.")
            vocab = []
        
        print("DEBUG: create_vocab restituisce un vocabolario con", len(vocab), "voci.")
        return vocab

    
    def create_framelabel(self):
        if 'train_iso_file' in self.dataset_cfg:
            iso_file = self.dataset_cfg['{}_iso_file'.format(self.split)]
            with open(iso_file, 'rb') as f:
                iso_ann = pickle.load(f)
            vfile2items = defaultdict(list)
            for item in iso_ann:
                vfile2items[item['video_file']].append(item)
            vfile2vlen = {}
            for item in self.annotation:
                vfile2vlen[item['name']] = item['num_frames']
            vfile2framelabel = {}
            for vfile in vfile2items.keys():
                vlen = vfile2vlen[vfile]
                label = ['<blank>']*vlen
                for item in vfile2items[vfile]:
                    start, end = item['start'], item['end']
                    for i in range(start, end):
                        label[i] = item['label']
                vfile2framelabel[vfile] = label
            return vfile2framelabel
        return {}
    
    def load_other_variants(self):
        other_vars = {}
        if self.dataset_cfg['dataset_name'] == 'WLASL_2000':
            others = ['1000', '300', '100']
            for o in others:
                other_vars[o] = {'dev': [], 'test': [], 'vocab_idx': []}
                with open(os.path.join(self.root, o+'.json'), 'rb') as f:
                    data = json.load(f)
                for k,v in data.items():
                    split = v['subset']
                    if split=='val':
                        other_vars[o]['dev'].append(k)
                    elif split=='test':
                        other_vars[o]['test'].append(k)
                        label = v['label']
                        label_idx = self.vocab.index(label)
                        if label_idx not in other_vars[o]['vocab_idx']:
                            other_vars[o]['vocab_idx'].append(label_idx)
        
        elif self.dataset_cfg['dataset_name'] == 'MSASL_1000':
            others = ['500', '200', '100']
            for o in others:
                other_vars[o] = {'dev': [], 'test': [], 'vocab_idx': [_ for _ in range(int(o))]}
                if self.split != 'train':
                    for item in self.annotation:
                        if self.vocab.index(item['label']) < int(o):
                            other_vars[o][self.split].append(item['name'])

        return other_vars

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        return self.annotation[idx]


def build_dataset(dataset_cfg, split, task='ISLR'):
    dataset = ISLRDataset(dataset_cfg, split, task)
    return dataset
