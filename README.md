# ollamaServer

**How to use it**

1. Firstly, use `pip install -r requirements.txt`
2. Then, `python ollama.py --csv-file <path to test_for_student> --root <path to hw3_16fpv> --save-dir <path to output> --frames <frames for prediction> --work-num <processes for connect> --url <server url> --model <pick up model> --size <int> <int>`

**Attention**

1. save_dir 最好要以".csv"结尾
2. frames最好是16的因数，默认是4，大于16就等报错吧
3. work_num别搞那么多，没用的
4. url只用最基本的就行了，为了安全删了default
5. model可以切换为llama3.2-vision，直接在代码中更改就行了
6. llama3.2-vision 只支持1张图片，所以frames要设为1，不然会报错：status code 400
7. size 那里是连续两个整数
8. 懒鬼就直接`python ollama.py --url <url>`就得了
