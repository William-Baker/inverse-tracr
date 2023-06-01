
import contextlib
import faulthandler
import io
import os
import platform
import re
import signal
import tempfile
from enum import Enum
from typing import Any, Optional




# def unsafe_execute(
#     code_str: str,
#     func_name: Optional[str] = None,
#     args: Optional[dict[str, Any]] = None,
#     ground_truth: Optional[dict[tuple, Any]] = None,
#     timeout: float = 5.0,
#     debug: bool = False,
# ):
#     if len(code_str) == 0 or "def " not in code_str:
#         # No code found or no function found.
#         if debug:
#             print("No code found or no function found.", "\n", code_str)
#         return ExecResult(5)
#     func_match = re.findall(r"def (\w+)\s*\((.*?)\):", code_str)
#     if len(func_match) == 0:
#         # No proper function found in code.
#         if debug:
#             print("No proper function found in code.", "\n", code_str)
#         return ExecResult(5)
#     elif len(func_match) > 0 and func_name is None:
#         func_name = func_match[-1][0]
#     with outer_guard():
#         try:
#             with inner_guard(timeout):
#                 code_dct: dict = {}
#                 exec(code_str, code_dct)
#                 if ground_truth is None:
#                     if args is None:
#                         result = code_dct[func_name]()
#                     elif args is not None:
#                         result = code_dct[func_name](**args)

#                     # Multiprocessing.pool.map
#                     # (in utils.code_eval.pool_exec_processes())
#                     # cannot return 'generators'
#                     # (this may not catch all 'invalid' generator uses)
#                     if isinstance(result, range):
#                         result = list(result)

#                     return result
#                 elif ground_truth is not None:
#                     if all(
#                         [
#                             code_dct[func_name](*arguments) == res
#                             for arguments, res in ground_truth.items()
#                         ]
#                     ):
#                         return ExecResult(0)
#                     else:
#                         return ExecResult(1)
#         except Exception as e:
#             if debug:
#                 print(type(e), e, "\n", code_str)
#             if isinstance(e, TimeoutException):
#                 return ExecResult(2)
#             elif isinstance(e, SyntaxError):
#                 return ExecResult(3)
#             elif isinstance(e, TypeError):
#                 return ExecResult(4)
#             else:
#                 return ExecResult(5)



@contextlib.contextmanager
def guard_timeout(timeout):
    with time_limit(timeout) as timer:
        yield timer


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    if seconds <= 0.0:
        yield
    else:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        signal.signal(signal.SIGALRM, signal_handler)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)




class TimeoutException(Exception):
    pass



#%%


# try:
#     print("start")
#     with guard_timeout(200):
#         print("hi")
# except Exception as E:
#     if isinstance(E, TimeoutException):
#         print("timed out")
#     else:
#         print(E)
# print("end")

# #%%
# print("waited")

# with guard_timeout(10):
#         print("hi")
#         while(True):
#             pass

#%%