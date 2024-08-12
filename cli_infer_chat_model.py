import pickle
import sys

sys.path.append("./")
sys.path.append("./anygpt/src")
import os
import random
import argparse
import logging
import json
import traceback
from transformers import  GenerationConfig
from openai import OpenAI
import requests
import json
from datetime import datetime
from PIL import Image
import base64
import torchaudio
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_audio(audio_path, sample_rate, segment_duration=5, one_channel=True, start_from_begin=False):
    metadata = torchaudio.info(audio_path)
    num_frames = metadata.num_frames
    orig_sample_rate = metadata.sample_rate
    segment_length = segment_duration * orig_sample_rate if segment_duration != -1 else num_frames

    # 确定读取音频的起始位置
    start_frame = 0 if start_from_begin else random.randint(0, max(0, num_frames - segment_length))
    waveform, or_sample_rate = torchaudio.load(
        audio_path,
        frame_offset=start_frame,
        num_frames=segment_length,
    )

    # 重采样音频
    if or_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=or_sample_rate, new_freq=sample_rate)
        waveform = resampler(waveform)

    # 检查音频是否是双通道
    # 如果是多通道，求平均
    if one_channel and waveform.shape[0] >= 2:
        waveform = waveform.mean(dim=0).unsqueeze(0)
    
    return waveform


class AnyGPTChatInference:
    def __init__(
        self, 
        output_dir="infer_output/test",
    ):
        self.output_dir = output_dir
        self.music_sample_rate = 32000
        self.music_segment_duration = 5
        self.audio_sample_rate = 24000
        self.audio_segment_duration = 5
        self.history=""
        
    def lmdeploy_post_request(self, generation_config, preprocessed_prompts):
        client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')
        model_name = client.models.list().data[0].id

        data = {
            "model": model_name,
            "messages": preprocessed_prompts,
            "top_p": generation_config.top_p,
            "max_tokens": generation_config.max_new_tokens,
            "stop": ["<eos>"],
            "temperature":generation_config.temperature,
            # "repetition_penalty":generation_config.repetition_penalty
        }
        headers = {"User-Agent": "infer_demo", "Content-Type": "application/json"}
        response = requests.post(url = "http://0.0.0.0:23333/v1/chat/completions", headers=headers, json = data)
        return response

    def response(self, query, to_modality, instruction, voice_prompt):
        processed_query = r"[Human]:" + query + "<eoh>\n" + "[MMGPT]:"
        # print(f"preprocessed_prompts:{preprocessed_prompts}")
        messages = self.history + processed_query
        config_path='config/generate_config.json'
        if to_modality == "speech":
            config_path='config/speech_generate_config.json'
        elif to_modality == "music":
            config_path='config/music_generate_config.json'
        elif to_modality == "image":
            config_path='config/image_generate_config.json'
        config_dict = json.load(open(config_path, 'r'))
        generation_config = GenerationConfig(
            **config_dict
        )
        
        response = self.lmdeploy_post_request(generation_config, messages)
        # response = f'{chatbot_name}:' + response.json()["choices"][0]['message']['content']
        # response += "<eos>"
        # print(f"print_response:{response.json()}")
        content = response.json()["choices"][0]["message"]["content"]
        message = content["text"]
        self.history = content['history']
        print(f'history:{self.history}')
        if message != None:
            print(message)
        if 'file' not in content.keys():
            return
        file_type = content["file"]["type"]
        
        
        print("find {file_type}, saving......")
        file_data = content["file"]["content"]
        file_data = base64.b64decode(file_data.encode('utf-8'))
        self.save_file(file_type, file_data, instruction, voice_prompt)
        
        # print("extract_content_between_final_tags")
        # 将response写入文本
        # with open(os.path.join(self.output_dir, "response.txt"), 'a', encoding='utf-8') as f:
        #     f.write(response+'\n\n\n')
        # try:
        #     response = extract_content_between_final_tags(response, tag1=f"{chatbot_name}", tag2="<eom>").strip()
        # except:
        #     print(response)
        #     response = extract_content_between_final_tags(response, tag1=f"{chatbot_name}", tag2="<eos>").strip()
        # if to_modality != "text":
        #     self.postprocess(response, to_modality, instruction, voice_prompt)
        # return response
    
    def save_file(self, file_type, file_data, instruction, voice_prompt):
        now = datetime.now()
        if file_type == "image":
            generated_image = pickle.loads(file_data)
            filename = now.strftime("%m%d_%H%M%S") + '.jpg'
            generated_image.save(os.path.join(self.output_dir, instruction[:50]+filename))
            print("image saved: ", os.path.join(self.output_dir, instruction+filename))
        elif file_type == "speech":
            generated_wav, sample_rate = pickle.loads(file_data)
            filename = now.strftime("%m%d_%H%M%S") + '.wav' 
            if voice_prompt:
                file_name = os.path.join(self.output_dir, instruction[:20] + "--" +
                                        os.path.basename(voice_prompt) + "--" + filename)
            else:
                file_name = os.path.join(self.output_dir, instruction[:50]+filename)
            print("speech saved: ", file_name)
            torchaudio.save(file_name, generated_wav, sample_rate)
        elif file_type == "music":
            generated_music = pickle.loads(file_data)
            filename = now.strftime("%m%d_%H%M%S") + '.wav'
            file_name = os.path.join(self.output_dir, instruction[:50]+filename)
            print("music saved: ", file_name)
            torchaudio.save(file_name, generated_music, self.music_sample_rate)
        elif file_type == "audio":
            now = datetime.now()
            os.path.join(self.output_dir, instruction[:50]+now.strftime("%m%d_%H%M%S") + '.txt')
            # with open(os.path.join(self.output_dir, input_data[:50]+now.strftime("%m%d_%H%M%S") + '.txt'), 'w') as f:
            #     f.write(modality_content)
            generated_audio = pickle.loads(file_data)
            filename = now.strftime("%m%d_%H%M%S") + '.wav'
            file_name = os.path.join(self.output_dir, instruction[:50]+filename)
            print("saved: ", file_name)
            torchaudio.save(file_name, generated_audio, self.audio_sample_rate)
    def forward(
        self, 
        prompts
    ):
        inputs = prompts.split("|")
        instruction = inputs[1].strip()
        try:
            voice_prompt = inputs[4].strip()
        except:
            voice_prompt = ""
        if instruction == "clear":
            self.history = ""
            print("clear conversation history successfully!")
            return
        try:
            image_paths = inputs[3].strip().split(",")
            image_str_list = []
            for image_path in image_paths:
                image_pil = Image.open(image_path).convert('RGB')
                dumps_data = pickle.dumps(image_pil)
                serialized_str = base64.b64encode(dumps_data).decode('utf-8')
                image_str_list.append(serialized_str)
            inputs[3] = ",".join(image_str_list)
        except:
            pass
        try:
            speech_paths = inputs[5].strip().split(",")
            speech_str_list = []
            for speech_path in speech_paths:
                wav, sr = torchaudio.load(speech_path)
                dumps_data = pickle.dumps((wav, sr))
                serialized_str = base64.b64encode(dumps_data).decode('utf-8')
                speech_str_list.append(serialized_str)
            inputs[5] = ",".join(speech_str_list)
        except:
            pass
        try:
            music_paths = inputs[6].strip().split(",")
            music_str_list = []
            for music_path in music_paths:
                if isinstance(music_path, (list, tuple)):
                    waveform = [ load_audio(p, self.sample_rate, self.segment_duration, 
                                one_channel=True, start_from_begin=True).numpy() for p in music_path]
                else:
                    waveform = load_audio(music_path, self.sample_rate, segment_duration=self.segment_duration, one_channel=True).squeeze(0)
                dumps_data = pickle.dumps(waveform)
                serialized_str = base64.b64encode(dumps_data).decode('utf-8')
                music_str_list.append(serialized_str)
            inputs[6] = ",".join(music_str_list)
        except:
            pass


        query = "|".join(inputs)
        try:
            to_modality = inputs[2].strip()
        except:
            to_modality = "text"

        response = self.response(query, to_modality, instruction, voice_prompt)
        
    def __call__(self, input):
        return self.forward(input)

    def interact(self):

        prompt = str(input(f"Please talk with AnyGPT chat:\n"))
        
        while prompt != "quit":
            try:
                self.forward(prompt)
            except Exception as e:
                traceback.print_exc()
                print(e)
            prompt = str(input(f"Please input prompts:\n"))
            
            
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="infer_output/test")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    infer = AnyGPTChatInference(
        args.output_dir
    )

    infer.interact()