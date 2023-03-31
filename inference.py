import os
import sys
import time
import torch
from autograd_4bit import load_llama_model_4bit_low_ram, Autograd4bitQuantLinear
config_path = './llama-13b-4bit/'
model_path = './llama-13b-4bit.pt'
model, tokenizer = load_llama_model_4bit_low_ram(config_path, model_path, groupsize=-1)

print('Fitting 4bit scales and zeros to half')
model.half()
for n, m in model.named_modules():
    if isinstance(m, Autograd4bitQuantLinear):
        if m.groupsize == -1:
            m.zeros = m.zeros.half()
        m.scales = m.scales.half()
        m.bias = m.bias.half()

print('Apply AMP Wrapper ...')
from amp_wrapper import AMPWrapper
wrapper = AMPWrapper(model)
wrapper.apply_generate()

prompt = '''Basic explanation of log4shell:'''
batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
batch = {k: v.cuda() for k, v in batch.items()}

start = time.time()
with torch.no_grad():
    generated = model.generate(inputs=batch["input_ids"],
                               do_sample=True,
                               use_cache=True,
                               repetition_penalty=1.1,
                               max_new_tokens=2000,
                               temperature=0.5,
                               top_p=0.9,
                               top_k=0,
                               return_dict_in_generate=True,
                               output_attentions=False,
                               num_beams=1,
                               no_repeat_ngram_size=0,
                               output_hidden_states=False,
                               early_stopping=False,
                               length_penalty=1,
                               min_length=0,
                               output_scores=False)

print('Generating...')
result_text = tokenizer.decode(generated['sequences'].cpu().tolist()[0])


end = time.time()
print(result_text)
print("gen time is: ")
print(end - start)
