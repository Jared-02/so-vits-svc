import os
import argparse
import re

from tqdm import tqdm
from random import shuffle
import json
import wave

config_template = json.load(open("configs_template/config_template.json"))

pattern = re.compile(r'^[\.a-zA-Z0-9_\/]+$')

def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # 获取音频帧数
        n_frames = wav_file.getnframes()
        # 获取采样率
        framerate = wav_file.getframerate()
        # 计算时长（秒）
        duration = n_frames / float(framerate)
    return duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="./filelists/train.txt", help="path to train list")
    parser.add_argument("--val_list", type=str, default="./filelists/val.txt", help="path to val list")
    parser.add_argument("--source_dir", type=str, default="./dataset/44k", help="path to source dir")
    args = parser.parse_args()
    
    rootpath = args.source_dir

    train = []
    val = []
    idx = 0
    spk_dict = {}
    spk_id = 0
    for speaker in tqdm(os.listdir(rootpath)):
        items = []
        spk_dict[speaker] = spk_id
        for file in os.listdir(f"{rootpath}/{speaker}"):
            if file.endswith(".wav"):
                file = file[:-4]
                wave_path = f"{rootpath}/{speaker}/{file}.wav"
                spec_path = f"{rootpath}/{speaker}/{file}.spec.pt"
                soft_path = f"{rootpath}/{speaker}/{file}.wav.soft.pt"
                f0_path = f"{rootpath}/{speaker}/{file}.wav.f0.npy"
                assert os.path.isfile(wave_path), wave_path
                items.append(
                    f"{spk_id}|{wave_path}|{spec_path}|{soft_path}|{f0_path}")
        spk_id += 1
        shuffle(items)
        train += items[2:]
        val += items[:2]

    shuffle(train)
    shuffle(val)
            
    print("Writing", args.train_list)
    with open(args.train_list, "w") as f:
        for line in tqdm(train):
            item_paths = line
            f.write(item_paths + "\n")
        
    print("Writing", args.val_list)
    with open(args.val_list, "w") as f:
        for line in tqdm(val):
            item_paths = line
            f.write(item_paths + "\n")

    config_template["spk"] = spk_dict
    config_template["model"]["n_speakers"] = spk_id
	
    print("Writing configs/config.json")
    with open("configs/config.json", "w") as f:
        json.dump(config_template, f, indent=2)
