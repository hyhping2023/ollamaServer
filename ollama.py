import requests
import json
import os, io, glob
import pandas as pd
import base64, time
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from multiprocessing import Pool

class Dataloader(Dataset):
    def __init__(self, csv_file='./test_for_student.csv', 
                 frames = 1, root='hw3_16fpv', already = None):
        self.df = pd.read_csv(csv_file, header=None, skiprows=1)[0].tolist()
        if already is not None:
            self.dfal = pd.read_csv(already, header=None, skiprows=1)
            self.retry = self.dfal[0][self.dfal.where(self.dfal[1] == -1).dropna().index]
            self.miss = set(self.df) - set(self.dfal[0].tolist())
            self.df = self.retry.tolist() + list(self.miss)
        self.root = root
        self.frames = frames

    def __getitem__(self, index):
        path = os.path.join(self.root, self.df[index]+'.mp4')
        files = sorted(glob.glob(os.path.join(path, '*.jpg')))
        return self.df[index], [files[i] for i in range(0, len(files), len(files)//self.frames)]

    def __len__(self):
        return len(self.df)

def png2base64(img_path, size=(224,224)):
    with Image.open(img_path) as img:
        resized_img = img.resize(size)
        # 将图片保存到一个字节流中
        img_byte_arr = io.BytesIO()
        resized_img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode('utf-8')
    
def getIndex(text:str):
    index = -1
    if text.find("[[[") != -1:
        index = int(text[text.find("[[[")+3])
    else:
        for i in text:
            if i.isdigit():
                index = int(i)
                break
    return index

def request(url, name:str, data_list:list, prompt, model, size=(224, 224),
            max_tokens=100, max_retries=5, temperature=0.8, top_p=0.8, top_k=5):
    # print(name)
    encoded_data = [png2base64(data, size) for data in data_list]
    data = {
        "model": model,
        "prompt": prompt,
        "stream":False,
        "images": encoded_data,
        "options":{
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
        }
        }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers, stream=False)
    print(response.json())

    retries = 0
    while response.status_code != 200 and retries < max_retries:
        retries += 1
        print("Retrying for {}th time. Retries left: {}. Name: {}. Status code: {}".format(retries, max_retries-retries, name, response.status_code), end='\r')
        time.sleep(1)
        response = requests.post(url, data=json.dumps(data), headers=headers, stream=False)

    if response.status_code == 200:
        try:
            output = response.json()['response']
        finally:
            response.close()
    else:
        print("\nerror")
        return name, "-114514"

    return name, output

class Client:
    def __init__(self, url, categories, prompt:str, worker_num=6, model='llava:13b', size=(224,224)):
        self.url = url
        self.categories = categories
        self.pool = Pool(worker_num)
        self.worker_num = worker_num
        self.workers = []
        self.prompt = prompt
        self.model = model
        self.size = size

    def start(self, dataloader: DataLoader):
        for name, data in tqdm(dataloader):
            while len(self.pool._cache) >= self.worker_num:
                time.sleep(0.2)
            # self.workers.append(self.pool.apply_async(request, (self.url, name, data, self.prompt, self.model, self.size)))
            print(self.url)
            request(self.url, name, data, self.prompt, self.model, self.size)
        self.pool.close()
        self.pool.join()
        results = {}
        for worker in self.workers:
            name, response = worker.get()
            result = getIndex(response)
            results[name] = result
        return results

    def save_result(self, results, save_dir='output.csv', csv_file='./test_for_student.csv'):
        temp = Dataloader(csv_file=csv_file, frames = 6, root='hw3_16fpv')
        with open(save_dir, "w") as f:
            f.writelines("Id,Category\n")
            for name in temp.df:
                f.writelines(f"{name},{results[name]}\n")

def makePrompt(categories:dict):
    prompt = '''
    Please classify the picture scene according to the picture. 
    You should carefuuly analyze what prople in the scene are really doing something and understand what they truly want to do.\n
    '''+"\n".join([f"If the scene is about {v}, the category index is {k}" for k,v in categories.items()])+\
    "\nOnly the catefory index is needed. You should return in the form of [[[index]]]."

    return prompt

categories  = {
    "0":"play basketball",
    "1":"mowing the lawn",
    "2":"play the guitar",
    "3":"play the piano",
    "4":"play the drum",
    "5":"pen beat",
    '6':'brithday',
    "7":"singing",
    "8":"scratch an itch",
    "9":"remove the snow",
}

url = 'http://{}:11434/api/generate'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", default="./test_for_student.csv")
    parser.add_argument("--root", default="../hw3_16fpv")
    parser.add_argument("--save-dir", default="./output.csv")
    parser.add_argument("--frames", default=4, type=int)
    parser.add_argument("--worker-num", default=6, type=int)
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--model", choices=['llava:13b', 'llama3.2-vision'], default='llava:13b')
    parser.add_argument("--size", type=int, nargs=2, default=(224, 224))
    args = parser.parse_args()

    prompt = makePrompt(categories)

    dataloader = Dataloader(csv_file=args.csv_file, frames=args.frames, root=args.root)
    client = Client(url.format(args.url), categories, prompt, worker_num=args.worker_num, model=args.model, size=args.size)
    results = client.start(dataloader)
    client.save_result(results, args.save_dir, args.csv_file)
