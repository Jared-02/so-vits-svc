import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import modules.commons as commons
import utils
from modules.mel_processing import spectrogram_torch, spec_to_mel_torch
from utils import load_wav_to_torch, load_filepaths_and_text

# import h5py


"""Multi speaker version"""


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1. 加载音频文件、说话人 ID 和声学特征；
        2. 过滤音频信号，根据采样率 (hop_length) 计算其采样帧数，用于划分 bucket；
        3. 对音高 (f0) 进行插值，并转换为 PyTorch 张量；
        4. 对齐多个声学特征的长度；
        5. 返回包含以下元素的元组：
           - 内容提取器特征输入
           - 音高 (f0)
           - 线性谱 (spec)
           - 归一化音频信号张量
           - 说话人 ID
           - 声学特征的有声/无声分类标签

        可以根据长度筛选数据实例，在加载到内存后，可以高效地访问数据实例，用于训练或其他处理任务。
    """

    def __init__(self, audiopaths, hparams, all_in_mem: bool = False):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.sampling_rate = hparams.data.sampling_rate
        self.use_sr = hparams.train.use_sr
        self.spec_len = hparams.train.max_speclen
        self.spk_map = hparams.spk

        random.seed(1234)
        random.shuffle(self.audiopaths)
        self._filter()
        
        self.all_in_mem = all_in_mem
        if self.all_in_mem:
            self.cache = [self.get_items_pair(p) for p in self.audiopaths]

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_filtered = []
        lengths = []
        for spk, wave, spec, soft, f0 in self.audiopaths:
            audiopaths_filtered.append([soft, f0, spec, wave, spk])
            lengths.append(os.path.getsize(wave) // (2 * self.hop_length))
        self.audiopaths = audiopaths_filtered
        self.lengths = lengths
    
    def get_items_pair(self, audiopaths):

        soft_path, f0_path, spec_path, wave_path, spk_id = audiopaths[:5]
        c, f0, uv = self.get_features(soft_path, f0_path)
        spec, audio_norm = self.get_audio(wave_path, spec_path)
        spk = self.get_sid(spk_id)

        # Ensure c and spec have similar lengths
        # truncate arrays to match the smaller length
        lmin = min(c.size(-1), spec.size(-1))
        assert abs(c.size(-1) - spec.size(-1)) < 3, (c.size(-1), spec.size(-1), f0.shape, wave_path)
        assert abs(audio_norm.shape[1]-lmin * self.hop_length) < 3 * self.hop_length
        spec, c, f0, uv = spec[:, :lmin], c[:, :lmin], f0[:lmin], uv[:lmin]
        audio_norm = audio_norm[:, :lmin * self.hop_length]

        return c, f0, spec, audio_norm, spk, uv

    def get_audio(self, filename, spec_path):
        # normalized path format
        filename = filename.replace("\\", "/")
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = spec_path

        # Ideally, all data generated after Mar 25 should have .spec.pt
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                                     self.sampling_rate, self.hop_length, self.win_length,
                                     center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_features(self, soft_path, f0_path):
        f0 = np.load(f0_path)
        f0, uv = utils.interpolate_f0(f0)
        f0 = torch.FloatTensor(f0)
        uv = torch.FloatTensor(uv)

        c = torch.load(soft_path)
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[0])
        return c, f0, uv
    
    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        if self.all_in_mem:
            return self.cache[index]
        else:
            return self.get_items_pair(self.audiopaths[index])

    def __len__(self):
        return len(self.audiopaths)


class TextAudioCollate:

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[1] for x in batch]),
            dim=0, descending=True)

        max_c_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])

        lengths = torch.LongTensor(len(batch))

        c_padded = torch.FloatTensor(len(batch), batch[0][0].shape[0], max_c_len)
        f0_padded = torch.FloatTensor(len(batch), max_c_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][2].shape[0], max_c_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        spkids = torch.LongTensor(len(batch), 1)
        uv_padded = torch.FloatTensor(len(batch), max_c_len)

        c_padded.zero_()
        spec_padded.zero_()
        f0_padded.zero_()
        wav_padded.zero_()
        uv_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            c = row[0]
            c_padded[i, :, :c.size(1)] = c
            lengths[i] = c.size(1)

            f0 = row[1]
            f0_padded[i, :f0.size(0)] = f0

            spec = row[2]
            spec_padded[i, :, :spec.size(1)] = spec

            wav = row[3]
            wav_padded[i, :, :wav.size(1)] = wav

            spkids[i, 0] = row[4]

            uv = row[5]
            uv_padded[i, :uv.size(0)] = uv

        return c_padded, f0_padded, spec_padded, wav_padded, spkids, lengths, uv_padded

class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    这是一个继承自 torch.utils.data.distributed.DistributedSampler 的 Python 类，使得同一个 batch 中多个输入音频的长度相似。长度组通过边界指定，例如 [b1, b2, b3]，每个 batch 都包含于 {x | b1 < length(x) <= b2} 或者 {x | b2 < length(x) <= b3} 中。它会丢弃不在边界内的样本，例如当边界为 [b1, b2, b3] 时，任何 length(x) <= b1 或 length(x) > b3 的数据将被忽略。

    init(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True) ：

    参数：

    dataset: 继承自 torch.utils.data.Dataset 的数据集
    batch_size: int，每个 batch 大小
    boundaries: list of int，边界列表
    num_replicas: int or None，默认为 None，分布式训练时参与训练的 GPU 数
    rank: int or None，默认为 None，当前 GPU ID
    shuffle：bool，默认为 True，是否在 epoch 开始时进行洗牌

    方法：

    iter(self)：返回一个可迭代的 batch，其中每个 batch 的数据长度范围都在边界内。
    len(self)：返回数据集中样本数量除以 batch 大小得到的结果值。
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        # 根据边界 (boundary) 创建桶 (bucket)，再根据长度 (length) 划分
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            # 补余
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
        """
        使用二分法划分
        """
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
