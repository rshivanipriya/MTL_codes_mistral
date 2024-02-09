
#import intel_extension_for_pytorch as ipex
import torch
import time
import argparse

from bigdl.llm.transformers import AutoModelForCausalLM
from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1#instruction-format
MISTRAL_PROMPT_FORMAT = """<s>[INST]Issue:Provide a format to present in the court for below details. Subject: Society registration.[/INST]"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Mistral model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="mistralai/Mistral-7B-Instruct-v0.1",
                        help='The huggingface repo id for the Mistral (e.g. `mistralai/Mistral-7B-Instruct-v0.1` and `mistralai/Mistral-7B-v0.1`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path2= args.repo_id_or_model_path
    #model_path1 = args.repo_id_or_model_path
    model_path = 'c:/Users/Administrator/llm/'
    model_name = "mistralai/Mistral-7B-v0.1"

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    # model = AutoModelForCausalLM.from_pretrained(model_path2,load_in_4bit=True,trust_remote_code=True)
    #model = AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True)
    #model_path = model_path + model_name+"-int4"
    #model = AutoModel.load_low_bit(model_path2, trust_remote_code=True,optimize_model=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path2,trust_remote_code=True)
    #model.save_low_bit(model_path)
    model = AutoModelForCausalLM.load_low_bit(model_path)
    #model.save_4bit(model_path)
   # model=ipex._optimize_transformers(model,dtype=amp_type,inplace=True)
    #model.save_pretrained('c:/Users/Administrator/llm') 
    # Generate predicted tokens
    with torch.inference_mode():
        prompt = MISTRAL_PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        st = time.time()
        # if your selected model is capable of utilizing previous key/value attentions
        # to enhance decoding speed, but has `"use_cache": false` in its model config,
        # it is important to set `use_cache=True` explicitly in the `generate` function
        # to obtain optimal performance with BigDL-LLM INT4 optimizations
        output = model.generate(input_ids,
                                max_new_tokens=3000)
        end = time.time()
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Output', '-'*20)
        print(output_str)
