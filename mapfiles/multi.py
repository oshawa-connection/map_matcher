import multiprocessing
from multiprocessing import Value
  
error_value = Value('i',0)  #assign as integer type
  
def get_data(url):
    try:
        raise ValueError
  
    except ValueError as e:
       #if error thrown up simply count it
        error_value.acquire()
        error_value.value += 1
        error_value.release()
        pass # <- not required, does nothing
  
  
if __name__ == '__main__':
  
# ---- Perform multi-processing 
    p = multiprocessing.Pool(6)
    results = p.map(get_data,'argslist')
    p.close()
    p.join()
    print(error_value.value)
    if (error_value.value > 6):
        print('hello world')