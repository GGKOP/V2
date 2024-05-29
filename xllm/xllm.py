from typing import List, Optional, Union

from xllm.engine import XLLMEngine
from xllm.engine import EngineArgs


class Request:
    def __init__(self, worker_id:int, prompt:str)->None:
        self.worker_id = worker_id
        self.prompt = prompt

class XLLM:
    """An LLM for generating texts from given prompts"""

    def __init__(
            self,
            model: str,
    ) -> None:
         engine_args = EngineArgs(
              model =model,
         )
         self.xllm_engine = XLLMEngine.from_engine_args(engine_args)        
         
    def add_request(
            self,
            prompt:Optional[str],
        ) ->None:
            request = Request(prompt=prompt, ID=len(self.xllm_engine.requests))
            self.xllm_engine.add_request(request)    
        
            if prompts is None:
                prompts = []
            elif isinstance(prompts, str):
                prompts = [prompts]
            for prompt in prompts:
                self._add_request(prompt)
            return self.xllm_engine.generate()
    

    def _run_engine(self)->None:
        self.xllm_engine.run()


