# ollamaServer

**Goal**

To help goog brother in HKUST(GZ) to use open source LVLM to solve AIAA2205 homework 3. This code is specific for ollama. If you have enough money, you can also use API provided by OpenAI anywhere.

**Function**

* Predict.py This is used to categorize the photos
* preLabel.py (Optional)This is used to analyze the category based on the training set.

**File Stricture**

```
.
├── README.md
├── descriptions.txt
├── preLabel.py
├── predict.py
├── requirements.txt
├── results.txt
├── test_for_student.csv
└── trainval.csv
```

**How to use it**

1. Firstly, use `pip install -r requirements.txt`
2. (Optional) You can use preLabel.py to generate label and get categories information by `python ollama.py --csv-file <path to trainval.csv> --root <path to hw3_16fpv> --frames <frames for prediction> --work-num <processes for connect> --url <server url> --model <pick up a model> --size <int> <int> --offset <frame offset> --sample <number of sampling>`
3. Then, `python ollama.py --csv-file <path to test_for_student> --root <path to hw3_16fpv> --save-dir <path to output> --frames <frames for prediction> --work-num <processes for connect> --url <server url> --model <pick up model> --size <int> <int> --offset <frame offset> --fix`

**Attention**

1. save_dir 最好要以".csv"结尾。
2. frames最好是16的因数，默认是4，大于16就等报错吧。
3. work_num别搞那么多，没用的，代表的是同时发起的请求数量。
4. url只用最基本的就行了，为了安全删了default。
5. model可以切换为llama3.2-vision，直接加上`--model <model>`。
6. llama3.2-vision 只支持1张图片，所以frames要设为1，不然会报错：status code 400 bad request。
7. size 那里是连续两个整数，代表图片裁切后大小。
8. offset是指在均匀抽取视频片段中每个连续帧组中帧的偏移量，代表选取组内不同的帧，offset的范围不能超过组的大小且要考虑末尾情况，否则会被设置为0。
9. sample代表在分析类别时从每个类别中抽取的样本数量
10. fix选项代表可以后期针对缺失的和错误的进行修复，这是可选项。
11. 懒鬼就直接`python ollama.py --url <url>`就得了，有需要在逐步加入参数。
