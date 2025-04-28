"""
Worker pool for parallel processing in Sapiens model

This module provides a worker pool implementation that can run functions
in parallel processes and handle exceptions properly.
"""

import functools
import multiprocessing as mp
import traceback as tb
from multiprocessing.pool import Pool


class AsyncWorkerExceptionsWrapper:
    """Wrapper for catching exceptions in worker processes"""
    
    def __init__(self, callable):
        """
        Initialize the wrapper
        
        Args:
            callable: Function to wrap
        """
        self.__callable = callable
        self._logger = mp.log_to_stderr()

    def __call__(self, *args, **kwargs):
        """
        Call the wrapped function and handle exceptions
        
        Args:
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function call
            
        Raises:
            Any exception raised by the function
        """
        try:
            result = self.__callable(*args, **kwargs)
        except Exception as e:
            self._logger.error(tb.format_exc())
            raise
        return result


class WorkerPool(Pool):
    """
    Worker pool that runs a function on each value that is put.
    
    This pool is designed so that if an exception is thrown in a child process,
    the main process will be notified and can handle it appropriately.
    """

    def __init__(self, func, *args, **kwargs):
        """
        Initialize the worker pool
        
        Args:
            func: Function to run on each item
            *args: Arguments to pass to Pool constructor
            **kwargs: Keyword arguments to pass to Pool constructor
        """
        self.func = func
        super().__init__(*args, **kwargs)

    def _result_collector(self, result):
        """
        Collects results from the pool and stores them in a list
        
        Args:
            result: The result of the function that was run on the pool
        """
        if isinstance(result, (list, tuple)):
            self.results.extend(result)
        else:
            self.results.append(result)

    def run(self, iterable, chunksize=1):
        """
        Run the function on each item in the iterable synchronously
        
        Args:
            iterable: Iterable of items to run func on
            chunksize: Number of items to run func on at once
            
        Returns:
            Results from the map operation
        """
        if all(isinstance(x, (list, tuple)) for x in iterable):
            results = self.starmap(self.func, iterable, chunksize)
        else:
            results = self.map(self.func, iterable)
        return results

    def run_async(self, iterable, chunksize=1):
        """
        Run the function on each item in the iterable asynchronously
        
        Args:
            iterable: Iterable of items to run func on
            chunksize: Number of items to run func on at once
            
        Returns:
            List to store results (will be populated as tasks complete)
        """
        self.results = []
        if all(isinstance(x, (list, tuple)) for x in iterable):
            self.starmap_async(
                AsyncWorkerExceptionsWrapper(self.func),
                iterable,
                chunksize,
                callback=self._result_collector,
            )
        else:
            self.map_async(
                AsyncWorkerExceptionsWrapper(self.func),
                iterable,
                chunksize,
                callback=self._result_collector,
            )
        return self.results

    def finish(self) -> None:
        """Shutdown the pool and clean-up threads"""
        self.close()
        self.join()