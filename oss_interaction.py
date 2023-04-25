import oss2
import os
import time
import tempfile
import secrets
import argparse
import yaml
import multiprocessing
from tqdm import tqdm

if multiprocessing.cpu_count() >= 8:
    threads_used = 6
else:
    threads_used = multiprocessing.cpu_count() - 2

def initialize(config_path: str):
    try:
        with open(config_path, 'r') as f:
            oss_config = yaml.load(f, Loader=yaml.FullLoader)
        accesskey_id = oss_config['AccessKey']['AccessKey_Id']
        accesskey_secrt = oss_config['AccessKey']['AccessKey_Secret']
        endpoint_domain = oss_config['Bucket']['Endpoint']
        bucket_name = oss_config['Bucket']['Bucket_Name']
        auth = oss2.Auth(accesskey_id, accesskey_secrt)
        bucket = oss2.Bucket(auth, endpoint_domain, bucket_name)
        return auth, bucket
    except FileNotFoundError:
        oss_config = dict(
            AccessKey={},
            Bucket={}
            )
        oss_config['AccessKey']['AccessKey_Id'] = 'yourAccessKeyId'
        oss_config['AccessKey']['AccessKey_Secret'] = 'yourAccessKeySecret'
        oss_config['Bucket']['Endpoint'] = 'https://oss-cn-hangzhou.aliyuncs.com'
        oss_config['Bucket']['Bucket_Name'] = 'examplebucket'
        with open(config_path, 'w') as f:
            yaml.dump(oss_config, f)
        print(f"Warning: Default config file has been created: {config_path}")
        return exit()

class TqdmUpTo(tqdm):
    # 参照 tqdm 4.65.0 文档 Hooks and callbacks 章节的回调参数进度条实现
    def update_to(self, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        return self.update(bsize - self.n)

def link_test_process():
    timestamp = int(time.time())
    random_number = secrets.token_hex(32)
    upload_filename = f"LinkTest_{timestamp}.txt"
    bucket.put_object(upload_filename, random_number)
    exist = bucket.object_exists(upload_filename)
    if exist:
        bucket.delete_object(upload_filename)
        print('link testing pass!')
    else:
        print('Error: link testing fail!')

def simple_upload(upload_file_localpath: str):
    # Python SDK 2.1.0+ 简单上传进度条实现（小于 5 GB）
    upload_filename = os.path.basename(upload_file_localpath)
    with open(upload_file_localpath, 'rb') as fileobj:
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024,
                      desc=upload_filename, miniters=1) as t:
            bucket.put_object(upload_filename, fileobj,
                              progress_callback=t.update_to)
            t.total = t.n

def resumable_upload(upload_file_localpath: str):
    # Python SDK 2.1.0+ 断点续传上传进度条实现（小于 48.8 TB）
    upload_filename = os.path.basename(upload_file_localpath)
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024,
                  desc=upload_filename, miniters=1) as t:
        oss2.defaults.connection_pool_size = threads_used
        oss2.resumable_upload(bucket, upload_filename, upload_file_localpath,
            store=oss2.ResumableStore(root=tempfile.gettempdir()),
            # 设置分片上传阈值 multipart_threshold。默认值为 10 MB。
            multipart_threshold=20480*1024,
            # 设置分片大小，单位为字节，取值范围为 100 KB~5 GB。默认值为 100 KB。
            part_size=1024*1024,
            # 设置上传回调进度函数。
            progress_callback=t.update_to,
            # 设置并发上传线程数 num_threads，需要将 oss2.defaults.connection_pool_size 设置为大于等于并发上传线程数。默认并发上传线程数为 1。
            num_threads=threads_used)
        t.total = t.n

def resumable_download(download_file: str, download_localpath:str):
    # Python SDK 2.1.0+ 断点续传下载进度条实现
    if download_localpath==None:
        download_localpath = os.path.join(os.path.expanduser('~'),
                                               'Downloads', download_file)
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024,
                  desc=download_file, miniters=1) as t:
        oss2.defaults.connection_pool_size = threads_used
        oss2.resumable_download(bucket, download_file, download_localpath,
            store=oss2.ResumableDownloadStore(root=tempfile.gettempdir()),
            # 设置分片下载阈值 multipart_threshold。默认值为 10 MB。
            multiget_threshold=20480*1024,
            # 设置分片大小，单位为字节，取值范围为100 KB~5 GB。默认值为100 KB。
            part_size=20480*1024,
            # 设置下载进度回调函数。
            progress_callback=t.update_to,
            # 如果使用num_threads设置并发下载线程数，请将oss2.defaults.connection_pool_size设置为大于或等于并发下载线程数。默认并发下载线程数为1。
            num_threads=threads_used)
        t.total = t.n

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, default="configs/oss_config.yaml", help='Aliyun OSS config file path')
    parser.add_argument('-t', '--link_test', action='store_true', default=False, help='Test the link to Aliyun OSS')
    parser.add_argument('-u', '--upload', type=str, help='Upload file local path')
    parser.add_argument('-d', '--download', type=str, default="dataset.zip", help='Download file oss path')
    parser.add_argument('-dp', '--download_path', type=str, default="dataset.zip", help='Download file local path')

    args = parser.parse_args()
    auth, bucket = initialize(args.config_path)
    if args.link_test:
        link_test_process()
    elif args.upload:
        resumable_upload(args.upload)
    elif args.download:
        try:
            resumable_download(args.download, args.download_path)
        except:
            resumable_download(args.download)