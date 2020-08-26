"""Distributed Task Scheduler"""
import os
import pickle
import logging
import sys
import distributed
from dask.distributed import Client

from distributed import progress
from dask import compute, persist
from warnings import warn
import multiprocessing as mp
from collections import OrderedDict

from .remote import RemoteManager
from .resource import DistributedResourceManager
from ..core import Task
from .reporter import *
from ..utils import AutoGluonWarning, AutoGluonEarlyStop, CustomProcess

logger = logging.getLogger(__name__)

__all__ = ['TaskSchedulerV1']


class TaskSchedulerV1(object):
    """Base Distributed Task Scheduler
    """
    PORT_ID = 8700
    def __init__(self, dist_ip_addrs=None):
        if dist_ip_addrs is None:
            dist_ip_addrs=[]
        if 'localhost' not in dist_ip_addrs:
            dist_ip_addrs.insert(0, 'localhost')
        # cluster = SSHCluster(dist_ip_addrs, scheduler_options={"port": self.PORT_ID})
        self._client = Client()
        self.scheduled_tasks = []
        self.finished_tasks = []
        print(self._client.scheduler_info()['services'])

    @classmethod
    def upload_files(cls, files, **kwargs):
        """Upload files to remote machines, so that they are accessible by import or load.
        """
        upload_jobs = []
        for client in self._clients:
            for file in files:
                upload_jobs.append(client.upload_file(file))
        # sync for uploads
        for job in upload_jobs:
            job.result()

    def _dict_from_task(self, task):
        if isinstance(task, Task):
            return {'TASK_ID': task.task_id, 'Args': task.args}
        else:
            assert isinstance(task, dict)
            return {'TASK_ID': task['TASK_ID'], 'Args': task['Args']}

    def add_task(self, task, **kwargs):
        """add_task() is now deprecated in favor of add_job().
        """
        warn("scheduler.add_task() is now deprecated in favor of scheduler.add_job().",
             AutoGluonWarning)
        self.add_job(task, **kwargs)

    def add_job(self, task, **kwargs):
        """Adding a training task to the scheduler.

        Args:
            task (:class:`autogluon.scheduler.Task`): a new training task

        Relevant entries in kwargs:
        - bracket: HB bracket to be used. Has been sampled in _promote_config
        - new_config: If True, task starts new config eval, otherwise it promotes
          a config (only if type == 'promotion')
        Only if new_config == False:
        - config_key: Internal key for config
        - resume_from: config promoted from this milestone
        - milestone: config promoted to this milestone (next from resume_from)
        """
        # adding the task
        # job = self._client.submit(lambda x:x, range(10))
        job = self._client.submit(self._run_dist_job, task.fn, task.args, task.resources.gpu_ids,
                                  resources={'process': task.resources.num_cpus,
                                             'GPU': task.resources.num_gpus})
        # job = self._client.submit(self._run_dist_job, task.fn, task.args, task.resources.gpu_ids)
        new_dict = self._dict_from_task(task)
        new_dict['Job'] = job
        self.scheduled_tasks.append(new_dict)
        return job

    def run_job(self, task):
        """Run a training task to the scheduler (Sync).
        """
        cls = TaskScheduler
        cls.resource_manager._request(task.resources)
        job = cls._start_distributed_job(task, cls.resource_manager)
        return job.result()

    @staticmethod
    def _run_dist_job(fn, args, gpu_ids):
        """Remote function Executing the task
        """
        if '_default_config' in args['args']:
            args['args'].pop('_default_config')

        if 'reporter' in args:
            pass

        if len(gpu_ids) > 0:
            # handle GPU devices
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpu_ids))
            os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = "0"

        try:
            ret = fn(**args)
        except AutoGluonEarlyStop:
            ret = None
        except Exception as e:
            ret = str(e)

        return ret

    def _cleaning_tasks(self):
        new_scheduled_tasks = []
        for task_dict in self.scheduled_tasks:
            if task_dict['Job'].done():
                self.finished_tasks.append(self._dict_from_task(task_dict))
            else:
                new_scheduled_tasks.append(task_dict)
        if len(new_scheduled_tasks) < len(self.scheduled_tasks):
            self.scheduled_tasks = new_scheduled_tasks
        logger.info(f'Num of Finished Tasks is {self.num_finished_tasks}')

    def join_tasks(self):
        warn("scheduler.join_tasks() is now deprecated in favor of scheduler.join_jobs().",
             AutoGluonWarning)
        self.join_jobs()

    def join_jobs(self, timeout=None):
        """Wait all scheduled jobs to finish
        """
        _jobs = [j['Job'] for j in self.scheduled_tasks]
        try:
            progress(persist(_jobs), fifo_timeout=None)
            logger.info('')
        except distributed.TimeoutError as e:
            logger.error(str(e))
        except:
            logger.error("Unexpected error:", sys.exc_info()[0])
            raise
        # for task_dict in self.scheduled_tasks:
        #     try:
        #         task_dict['Job'].result(timeout=timeout)
        #     except distributed.TimeoutError as e:
        #         logger.error(str(e))
        #     except:
        #         logger.error("Unexpected error:", sys.exc_info()[0])
        #         raise
        self._cleaning_tasks()

    def shutdown(self):
        """shutdown() is now deprecated in favor of :func:`autogluon.done`.
        """
        warn("scheduler.shutdown() is now deprecated in favor of autogluon.done().",
             AutoGluonWarning)
        self.join_jobs()

    def state_dict(self, destination=None):
        """Returns a dictionary containing a whole state of the Scheduler

        Examples
        --------
        >>> ag.save(scheduler.state_dict(), 'checkpoint.ag')
        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination['finished_tasks'] = pickle.dumps(self.finished_tasks)
        destination['TASK_ID'] = Task.TASK_ID.value
        return destination

    def load_state_dict(self, state_dict):
        """Load from the saved state dict.

        Examples
        --------
        >>> scheduler.load_state_dict(ag.load('checkpoint.ag'))
        """
        self.finished_tasks = pickle.loads(state_dict['finished_tasks'])
        Task.set_id(state_dict['TASK_ID'])
        logger.debug('\nLoading finished_tasks: {} '.format(self.finished_tasks))

    @property
    def num_finished_tasks(self):
        return len(self.finished_tasks)

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(\n' + \
            str(self.resource_manager) +')\n'
        return reprstr
