import types

import torch.multiprocessing as mp

from .task import EmptyTask, StopTask, Task

mp = mp.get_context('spawn')


class Worker(object):
    def __init__(self):
        pass

    def process(self, task):
        pass


class SimpleWorker(Worker):
    def __init__(self, work_fn, init_fn=None):
        super(SimpleWorker, self).__init__()
        self.work_fn = work_fn
        self.resource = None
        if init_fn is not None:
            self.resource = init_fn()

    def process(self, task):
        if not isinstance(task, Task):
            raise Exception("Input is not a Task: {}".format(type(task)))
        if isinstance(task, EmptyTask):
            return EmptyTask()
        if isinstance(task, StopTask):
            return StopTask()
        return self.work_fn(self.resource, task)


class SimpleWorkerProcess(mp.Process):
    def __init__(self, work_fn, init_fn, job_queue, result_queue,
                 curr_worker_num, next_worker_num, name=None):
        super(SimpleWorkerProcess, self).__init__(name=name)
        self.worker = SimpleWorker(work_fn, init_fn)
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.curr_worker_num = curr_worker_num
        self.next_worker_num = next_worker_num

    def run(self):
        try:
            while True:
                task = self.job_queue.get()
                if isinstance(task, StopTask):
                    with self.curr_worker_num.get_lock():
                        self.curr_worker_num.value -= 1
                    break
                result = self.worker.process(task)

                # the result can be:
                #   1. None, means do not want to output result
                #   2. Task, means output one result
                #   3. List or Generator, means output more than one results

                if result is None:
                    continue
                elif isinstance(result, Task):
                    self.result_queue.put(result)
                elif isinstance(result, (list, types.GeneratorType)):
                    for r in result:
                        self.result_queue.put(r)
                else:
                    raise Exception("Illegal output type: {}".format(
                        type(result)))

            if self.curr_worker_num.value == 0:
                for _ in range(self.next_worker_num.value):
                    self.result_queue.put(StopTask())

        except (EOFError, FileNotFoundError, BrokenPipeError):
            pass
