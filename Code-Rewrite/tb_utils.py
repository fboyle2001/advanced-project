"""
Automatically launches and terminates the TensorBoard instance
Credit: https://stackoverflow.com/a/60021949 by Mano
"""

from multiprocessing import Process
import sys
import os

class TensorboardSupervisor:
    def __init__(self, log_dp):
            self.server = TensorboardServer(log_dp)
            self.server.start()
            print("Started Tensorboard Server")
            self.chrome = ChromeProcess()
            print("Started Chrome Browser")
            self.chrome.start()

    def finalize(self):
        if self.server.is_alive():
            print('Killing Tensorboard Server')
            self.server.terminate()
            self.server.join()
        # As a preference, we leave chrome open - but this may be amended similar to the method above

class TensorboardServer(Process):
    def __init__(self, log_dp):
        super().__init__()
        self.os_name = os.name
        self.log_dp = str(log_dp)
        # self.daemon = True

    def run(self):
        if self.os_name == 'nt':  # Windows
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self.log_dp}" 2> NUL')
        elif self.os_name == 'posix':  # Linux
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self.log_dp}" '
                      f'--host `hostname -I` >/dev/null 2>&1')
        else:
            raise NotImplementedError(f'No support for OS : {self.os_name}')
    
class ChromeProcess(Process):
    def __init__(self):
        super().__init__()
        self.os_name = os.name
        self.daemon = True

    def run(self):
        if self.os_name == 'nt':  # Windows
            os.system(f'start chrome  http://localhost:6006/')
        elif self.os_name == 'posix':  # Linux
            os.system(f'google-chrome http://localhost:6006/')
        else:
            raise NotImplementedError(f'No support for OS : {self.os_name}')