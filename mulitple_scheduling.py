import os
import subprocess
import re
import time
from multiprocessing import Process, Queue, Value
from datetime import datetime
from ctypes import c_bool
import json

# Paths and configurations
whisper_models = [
    "whisper_model/ggml-base.en.bin", 
    "whisper_model/ggml-medium.en.bin",
    "whisper_model/ggml-small.en.bin",
    "whisper_model/ggml-tiny.en.bin"
]
gemma_models = [
    "llm_model/gemma2_Q2_K.gguf", 
    "llm_model/gemma2_Q4_K_M.gguf",
    "llm_model/gemma2_Q8_0.gguf",
    "llm_model/gemma2.gguf",
    "llm_model/Llama-3.2-1B_Q2_K.gguf",
    "llm_model/Llama-3.2-1B_Q4_K_M.gguf",
    "llm_model/Llama-3.2-1B_Q8_0.gguf",
    "llm_model/Llama-3.2-1B.gguf",
    "llm_model/Llama-3.2-3B_Q2_K.gguf",
    "llm_model/Llama-3.2-3B_Q4_K_M.gguf",
    "llm_model/Llama-3.2-3B_Q8_0.gguf",
    "llm_model/Llama-3.2-3B.gguf",
    "llm_model/llama3-Q2_K.gguf",
    "llm_model/llama3-Q4_k_M.gguf",
    "llm_model/llama3-Q8_0.gguf",
    "llm_model/llama3.gguf",
    "llm_model/phi-3-Q2_k.gguf",
    "llm_model/phi-3-Q4_K_M.gguf",
    "llm_model/phi-3-Q8_0.gguf",
    "llm_model/phi-3.gguf",
    "BitNet/models/bitnet_b1_58-large/ggml-model-i2_s.gguf",
    "BitNet/models/Llama3-8B-1.5B-100B-tokens/ggml-model-i2-s.gguf"
]
whisper_main_path = "whisper.cpp/main"
llama_cli_path = "llama.cpp/llama-cli"
audio_directory = "audio/squad_audio_questions"
output_directory = "results/"
throughput_log_file = "results/throughput_log.json"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Get the list of audio files
audio_files = [os.path.join(audio_directory, f) for f in os.listdir(audio_directory) if f.endswith('.wav')]

def append_stats_to_json(stats_data, throughput_log_file):
    try:
        if os.path.exists(throughput_log_file):
            with open(throughput_log_file, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        existing_data.append(stats_data)
        with open(throughput_log_file, 'w') as f:
            json.dump(existing_data, f, indent=4)
    except Exception as e:
        print(e)
        print(f"Error appending to JSON file: {e}")

def transcribe_audio(audio_queue, whisper_model, producer_done):
    """
    Producer process: Transcribes audio files and puts the output in the queue.
    """
    for audio_file in audio_files:
        audio_basename = os.path.basename(audio_file)
        print(f"Transcribing Audio File: {audio_basename}")
        transcript_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = time.time()
        result = subprocess.run(
            [whisper_main_path, '-m', whisper_model, '-f', audio_file],
            capture_output=True, text=True
        )
        transcription = re.sub(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]', '', result.stdout.strip())
        transcription_time = time.time() - start_time
        transcript_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        audio_queue.put((audio_basename, transcription, transcription_time, transcript_start, transcript_end))
        print(f"Transcription Done: {audio_basename} ({transcription_time:.2f}s)")
    producer_done.value = True
    print("producer finished!!!!")

def generate_response(audio_queue, gemma_model, whisper_model, processed_audio_count, producer_done):
    """
    Consumer process: Takes transcription from the queue and generates a response.
    """
    while True:
        if not audio_queue.empty():
            audio_basename, transcription, transcription_time, whisper_start, whisper_end = audio_queue.get()
            print(f"Generating Response for: {audio_basename} with Whisper model: {whisper_model} and Gemma model: {gemma_model}")
            llm_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            start_time = time.time()
            prompt = f"""Based on the following transcription, provide a detailed and specific answer:
            Transcription: {transcription}
            Answer:"""
            result = subprocess.run(
                [llama_cli_path, '-m', gemma_model, '-p', prompt, '-n', '50'],
                capture_output=True, text=True
            )
            response_time = time.time() - start_time
            llm_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            response = result.stdout.strip()

            # Save results with model pairing in the filename
            output_file = os.path.join(output_directory, f'results_{os.path.basename(whisper_model)}_{os.path.basename(gemma_model)}_{audio_basename}.txt')
            with open(output_file, 'w') as file:
                file.write(f"Audio File: {audio_basename}\n")
                file.write(f"Whisper Model: {whisper_model}\n")
                file.write(f"Gemma Model: {gemma_model}\n")
                file.write(f"Whisper Start: {whisper_start}\n")
                file.write(f"Whisper end: {whisper_end}\n")
                file.write(f"Transcription Time: {transcription_time:.2f} seconds\n")
                file.write(f"Transcription: {transcription}\n")
                file.write(f"LLM Start: {llm_start}\n")
                file.write(f"LLM end: {llm_end}\n")
                file.write(f"Response Time: {response_time:.2f} seconds\n")
                file.write(f"Response: {response}\n")
            print(f"Response Saved for: {audio_basename} ({response_time:.2f}s)")

            processed_audio_count.value += 1  # Increment the processed audio count

        else:
            if producer_done.value and audio_queue.empty():
                break
            time.sleep(0.1)

def main():
    # shared variable
    processed_audio_count = Value('i', 0)
    producer_done = Value(c_bool, False)
    for whisper_model in whisper_models:
        for gemma_model in gemma_models:
            print(f"Starting process for Whisper model: {whisper_model} and Gemma model: {gemma_model}")
            
            audio_queue = Queue()
            start_time = time.time()

            producer = Process(target=transcribe_audio, args=(audio_queue, whisper_model, producer_done))
            consumer = Process(target=generate_response, args=(audio_queue, gemma_model, whisper_model, processed_audio_count, producer_done))
            
            producer.start()
            consumer.start()
            producer.join()
            consumer.join()

            end_time = time.time()
            duration = end_time - start_time
            throughput = processed_audio_count.value / duration
            stats_data = {
                'Whisper Model': whisper_model,
                'LLM Model': gemma_model,
                'Start Time': start_time,
                'End Time': end_time,
                'Duration': duration,
                'Audio Processed': processed_audio_count.value,
                'Throughput(files/sec)': throughput
            }
            append_stats_to_json(stats_data, throughput_log_file)

            print(f"Finished processing for Whisper model: {whisper_model} and Gemma model: {gemma_model}\n")
            processed_audio_count.value = 0
            producer_done.value = False

if __name__ == "__main__":
    main()
