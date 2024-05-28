from typing import List, Optional, Union

from xllm.engine import XLLMEngine

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

    def generate(
        self,
        prompts: Optional[Union[str,List[str]]] = None,
    )->List[RequestOutput]:
        
         
        def _add_request(
            self,
            prompt:Optional[str],
        ) ->None:
         