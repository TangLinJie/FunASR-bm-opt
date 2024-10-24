from funasr import AutoModel
import time
import multiprocessing 

# setting
dev_id = 0                              ## bm1684x/bm1688 device id
input_path = "./20240711090630019.wav" # "./chuanda.wav" # "./vad_example.wav" #        ## input audio path

def process():
    # offline asr demo
    model = AutoModel(model="iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",    ## 语音识别模型
                    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",                             ## 语音端点检测模型
                    punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",                ## 标点恢复模型
                    dev_id=dev_id,
                    )
    # inference
    start_time = time.time()
    res = model.generate(input=input_path,
                        batch_size_s=300,
                        )
    end_time = time.time()
    print(res)
    print("generate time:", end_time-start_time)

ps = []
for _ in range(3):
    p = multiprocessing.Process(target=process)
    p.start()
    ps.append(p)

for p in ps:
    p.join()


