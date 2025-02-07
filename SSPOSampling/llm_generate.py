from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class textGenerator:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto",
                                                          low_cpu_mem_usage=True, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left', trust_remote_code=True)

    def generate(self, prompt: str, n=20, temperature=1.4, top_p=0.95, top_k=60, max_gen_tokens=32):
        encoded_input = self.tokenizer(prompt, return_tensors="pt", padding=True,
                                       truncation=False)  # max_length=10240, 
        input_ids = encoded_input.input_ids.cuda()
        attention_mask = encoded_input.attention_mask.cuda()
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                num_beams=n,
                max_new_tokens=max_gen_tokens,
                num_return_sequences=n,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=pad_token_id,
            )

        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        del input_ids
        del attention_mask
        del generated_ids
        torch.cuda.empty_cache()

        return outputs
