from typing import Awaitable
from transformers import AutoTokenizer, AutoModelForCausalLM

from coding.protocol import StreamCodeSynapse
#from langchain_huggingface.llms import HuggingFacePipeline


def miner_init(self):
    """
    Initializes the miner. This function is called once when the miner is created.
    """

    self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
    self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)#.cuda()


    
def miner_process(self, synapse: StreamCodeSynapse) -> Awaitable:
    """
    The miner process function is called every time the miner receives a request. This function should contain the main logic of the miner.
    """
    prompt = f'<｜fim▁begin｜>{synapse.query}<｜fim▁end｜>'

    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    outputs = self.model.generate(**inputs, max_length=1024)[0]
    synapse.completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)


    return synapse
