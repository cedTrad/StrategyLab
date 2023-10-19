import multiprocessing

shared_variable = multiprocessing.Value("f")
shared_variable.value = 0


class Process(multiprocessing.Process):
    
    def __init__(self, counter, lock):
        super(Process, self).__init__()
        self.counter = counter
        self.lock = lock
        
    def run(self):
        for i in range(10):
            with self.lock:
                self.counter.value += 1
            
            
def main():
    counter = multiprocessing.Value("i", lock=True)
    counter.value = 0
    
    lock = multiprocessing.Lock()
    
    
    processes = [Process(counter, lock) for i in range(4)]
    [p.start() for p in processes]
    [p.join() for p in processes]   # processes are done
    print(counter.value)
    

if __name__ == "__main__":
    main()
    