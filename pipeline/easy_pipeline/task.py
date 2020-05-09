import time


class Task(object):
    def __init__(self):
        pass


class EmptyTask(Task):
    def __init__(self):
        super(EmptyTask, self).__init__()


class StopTask(Task):
    def __init__(self):
        super(StopTask, self).__init__()


class ValuedTask(Task):

    def __init__(self, value):
        super().__init__()
        self.value = value
