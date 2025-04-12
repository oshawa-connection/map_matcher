import time
from pynput.keyboard import Key, Listener


shared = True

def on_press(key):
    global shared
    print('{0} pressed'.format(
        key))
    
    shared = False
    
listener = Listener(on_press=on_press)
listener.start()


while (shared):
    print('CALCULATING!')
    time.sleep(5)

print('done')

listener.stop()