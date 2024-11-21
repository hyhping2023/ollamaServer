import requests
import json
import os, io, glob
import pandas as pd
import base64, time
from tqdm import tqdm
import argparse
from PIL import Image
from multiprocessing import Pool
from ollama import png2base64, Dataloader, request

class Client:
    def __init__(self, url, worker_num=6, model='llava:13b', size=(224,224)):
        self.url = url
        # self.categories = categories
        self.pool = Pool(worker_num)
        self.pool2 = Pool(5)
        self.worker_num = worker_num
        self.workers = []
        self.workers2 = [] 
        self.model = model
        self.size = size

    def start(self, dataloader: Dataloader):
        for name, data in tqdm(dataloader):
            while len(self.pool._cache) >= self.worker_num:  # waiting for the processes in the pool and blocking the new ones
                time.sleep(0.2)
            self.workers.append(self.pool.apply_async(request, (self.url, name, data, self.makePrompt(), self.model, self.size)))
            # request(self.url, name, data, self.prompt, self.model, self.size)
        self.pool.close()
        self.pool.join()
        dataframe = dataloader.label
        descriptions = {}
        for worker in self.workers:
            name, response = worker.get()
            if dataframe[name] not in descriptions:
                descriptions[dataframe[name]] = []
            descriptions[dataframe[name]].append(name + ": " + response)
        
        # part for intergrating the descriptions
        results = {}
        for category, description in tqdm(descriptions.items()):
            while len(self.pool2._cache) >= 5:  # waiting for the processes in the pool and blocking the new ones
                time.sleep(0.2)
            self.workers2.append(self.pool2.apply_async(request, (self.url, category, [], self.makePrompt(1, description), self.model)))
            # name, response = request(self.url, category, [], self.makePrompt(1, description), self.model)
        self.pool2.close()
        self.pool2.join()
        for worker2 in self.workers2:
            category, response = worker2.get()
            results[category] = response
        return descriptions, results


    def save_result(self, descriptions:dict, results:dict):
        with open('descriptions.txt', "w") as f:
            for category, description in descriptions.items():
                f.writelines(f"{category}:\n")
                f.writelines("\n".join(description))
                f.writelines("\n\n\n")

        with open('results.txt', "w") as f:
            for category, result in results.items():
                f.writelines(f"{category}:\n")
                f.writelines(result)
                f.writelines("\n\n\n")

    def makePrompt(self, mode:int = 0, previous:list = []):
        if mode == 0:
            prompt = '''
            Please classify the picture scene according to the picture. 
            You should carefuuly analyze what prople in the scene are really doing something and understand what they truly want to do.
            Then, based on the analysis results, you should provide a detailed description of the scene.\n
            '''
        else:
            prompt = '''
            Please based on the previous description \n[Start of previous description]\n{}\n[End of previous description]\n, classify the picture scene according to the description and prvide a brief category of the scene that best matches all the description.\n
            '''.format("\n".join(previous))
        return prompt

url = 'http://{}:11434/api/generate'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", default="./trainval.csv")
    parser.add_argument("--root", default="../hw3_16fpv")
    parser.add_argument("--save-dir", default="./output.csv")
    parser.add_argument("--frames", default=4, type=int)
    parser.add_argument("--worker-num", default=6, type=int)
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--model", choices=['llava:13b', 'llama3.2-vision'], default='llava:13b')
    parser.add_argument("--size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--offset", default=0, type=int)
    parser.add_argument("--sample", default=10, type=int)
    args = parser.parse_args()

    dataloader = Dataloader(csv_file=args.csv_file, frames=args.frames, root=args.root, offset=args.offset, sample=args.sample)
    client = Client(url=url.format(args.url), worker_num=args.worker_num, model=args.model, size=args.size)
    descriptions, results = client.start(dataloader)
    client.save_result(descriptions, results)
