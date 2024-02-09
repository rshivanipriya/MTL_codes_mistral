
#import intel_extension_for_pytorch as ipex
import torch
import time
import argparse

from bigdl.llm.transformers import AutoModelForCausalLM
from bigdl.llm.transformers import AutoModel
from transformers import AutoTokenizer

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1#instruction-format
MISTRAL_PROMPT_FORMAT = """<s>[INST]Issue:l am unable to login to my computer. content: Speeding up a slow computer: Run fewer programs at the same time
Restart your computer Remove viruses and malware Free up hard disk space Verify windows system files Uninstall unnecessary programs Adjust windows visual effects Run a disk scan
Defragment your hard disk Reinstall windows Advanced steps Upgrade your hardware
Problem: An application is running slowly Solution 1 Close and reopen the application. Solution 2
Update the application. To do this, click the Help menu and look for an option to Check for Updates. If you don't find this option, another idea is to run an online search for application updates. Problem: An application is frozen Sometimes an application may become stuck,or frozen. When this happens, you won't be able to close the window or click any buttons within the application. Solution 1 Force quit the application. If a program has become completely
unresponsive, you can press and hold Ctri+Shift+Esc on your keyboard to open the Task Manager. You can then select the unresponsive application and click End task to close it.
Solution 2 Restart the computer.
If you are unable to force quit an application, restarting your computer will close all open apps.
Problem: The computer is frozen
Sometimes your computer may become completely
unresponsive, or frozen. When this happens, you won't be able to click anywhere on the screen, open or close applications, or access shut-down options.
Solution 1 Restart Windows Explorer. To do this, press and hold Ctri+Alt+Delete on your keyboard to open the Task Manager. Next, locate and select Windows Explorer from the Processes tab and click
Restart. If you're using Windows 8, you may need to click More Details at the bottom of the window to see the Processes tab.
Solution 2 Press and hold the Power button for 5-10 seconds.
This will force the computer to shut down. Solution 3 If the computer still won't shut down, you can unplug the power cable from the electrical outlet. If you're using a laptop, you may be able to remove the battery to force the computer to turn off.
Note : This solution should be your last resort after trying the other suggestions above.
Problem: The mouse/keyboard has stopped working Solution 1 lf you're using a wired mouse or keyboard, make sure it's correctly plugged in to the computer.
Solution 2 If you're using a wireless mouse or keyboard, make sure it is turned on and that its batteries are charged. Problem The screen is blank Solution 1
The computer may just be in Sleep mode. Simply click the mouse or press any key on the keyboard to wake it. Solution 2
Make sure the monitor is plugged in and turned on . Solution 3
Make sure the computer is plugged in and turned on.
Solution 4 If you're using a desktop computer, make sure the monitor cable is properly connected to the computer tower and the monitor. Problem: I can't hear the sound on my computer
Solution 1 Check the volume level. Click the audio button in the bottom-right corner of the screen to make sure the sound is turned on and the volume is up. Solution
2 Check the audio player controls. Many audio and video players will have their own separate audio controls. Make sure the sound is turned on and the volume is up in the player.
Solution 3 Check the cables.
Make sure external speakers are plugged in, turned on, and connected to the correct audio port or a USB port. If your computer has color-coded ports, the audio output port will usually be green . Solution 4Connect headphones to the computer to determine if you can hear sound from the headphones.
Problem: Printer trouble Printers are common source of trouble, but it can be fixed easily. Go to Control panel choose Devices and Printers. Right click on your printer and choose to remove it.If there is no answer in the content provided, then you use your knowledge and provide the solution[/INST]"""

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
    #model=model.to('xpu')
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
