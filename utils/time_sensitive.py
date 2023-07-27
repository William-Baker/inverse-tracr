import multiprocessing, cloudpickle

def time_sensitive(func: callable, timeout):
    def wrapped_func(val):
        try:
            ret = func()
            val.value = cloudpickle.dumps(ret)
        except Exception as E:
            print(f"Error in time sensitve function: {E}")


    manager = multiprocessing.Manager()
    val = manager.Value(bytes, cloudpickle.dumps(None))
    

    p = multiprocessing.Process(target=wrapped_func, args=[val])
    p.start()
    p.join(timeout)
    while p.is_alive(): # the process hasnt finished yet
        p.terminate()   # kill it
        p.join()        # delete the thread
        return None

    return cloudpickle.loads(val.value)