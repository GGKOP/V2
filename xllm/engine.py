from typing import List
from xllm  import Request

class XLLMEngine:
    def __init__(
        self,
        Scheduler,
        Worker,
    )->None:
        self._init_workers()
        self.scheduler =Scheduler()

    def _add_request(self, request: Request)->None:
        self.scheduler.add_request(request)  

    def _init_workers(self)->None:
        self.workers = []
        for i in range(self.scheduler_config.num_workers):
            worker = Worker(i, self.scheduler)
            self.workers.append(worker)

    def run(self)->None:
        while True:
            request = self.scheduler.get_next_request()
            if request is None:
                break
            worker = self.workers[request.worker_id]
            worker.run_request(request)

    def _init_scheduler(self)->None:
        self.scheduler.init_scheduler()