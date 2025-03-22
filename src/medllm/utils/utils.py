import time

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution Time of {func.__name__}: {end - start}")
        return result
    return wrapper

def load_list_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip().lower() for line in file]