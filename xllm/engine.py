


class XLLMEngine:
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    )->None:
        self.scheduler_config = scheduler_config
        self._init_workers()
        self._init_cache()
        self.scheduler =Scheduler(scheduler_config,cache_config)

        def _init_cache(self) ->None:
        def from_engine_args() ->None