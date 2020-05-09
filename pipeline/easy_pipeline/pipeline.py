from .task import StopTask
from .worker import SimpleWorkerProcess, mp


class Pipeline(object):
    def __init__(self):
        pass

    def process(self, task):
        pass

    def start(self):
        pass


class PipelineItem(object):
    def __init__(self, work_fn, init_fn, worker_num, result_max_length=-1,
                 name=None):
        self.work_fn = work_fn
        self.init_fn = init_fn
        self.worker_num = worker_num
        self.result_max_length = result_max_length
        self.name = name


class SimplePipeline(object):
    def __init__(self, items, job_max_length=-1):
        super(SimplePipeline, self).__init__()
        self.manager = mp.Manager()
        self.job_queue = self.manager.Queue(job_max_length)
        self.process_pool = []
        self.result_queue = None
        job_queue = self.job_queue
        for idx in range(len(items)):
            pipeline_item = items[idx]
            task_process_pool = []

            # curr and next worker_num will be used to stop processes safely
            curr_worker_num = mp.Value('i', pipeline_item.worker_num)
            if idx < len(items) - 1:
                next_worker_num = mp.Value('i', items[idx + 1].worker_num)
            else:
                next_worker_num = mp.Value('i', 1)

            result_queue = self.manager.Queue(pipeline_item.result_max_length)

            for i in range(pipeline_item.worker_num):
                task_process_pool.append(
                    SimpleWorkerProcess(
                        pipeline_item.work_fn,
                        pipeline_item.init_fn,
                        job_queue,
                        result_queue,
                        curr_worker_num,
                        next_worker_num,
                        '%s-%d' % (pipeline_item.name, i)
                    ))
            job_queue = result_queue
            self.process_pool.append(task_process_pool)
        self.result_queue = result_queue

    def start(self):
        for task_process_pool in self.process_pool:
            for task_process in task_process_pool:
                task_process.start()

    def stop(self):
        if len(self.process_pool) > 0:
            task_process_pool = self.process_pool[0]
            for task_process in task_process_pool:
                task_process.job_queue.put(StopTask())

    def get_result_queue(self):
        return self.result_queue

    def join(self, timeout=1):
        for pool in self.process_pool:
            for process in pool:
                process.join(timeout)

    def terminate(self):
        for pool in self.process_pool:
            for process in pool:
                process.terminate()

    def close(self):
        self.join()
        self.terminate()
