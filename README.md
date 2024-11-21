# ollamaServer

**How to use it**

1. Firstly, use `pip install -r requirements.txt`
2. Then, `python ollama.py --csv-file <path to test_for_student> --root <path to hw3_16fpv> --save-dir <path to output> --frames <frames for prediction> --work-num <processes for connect> --url <server url>`

**Attention**

1. save_dir 最好要以".csv"结尾
2. frames最好是16的因数，默认是4，大于16就等报错吧
3. work_num别搞那么多，没用的
4. url只用最基本的就行了，为了安全删了default
5. 懒鬼就直接`python ollama.py --url <url>`就得了
