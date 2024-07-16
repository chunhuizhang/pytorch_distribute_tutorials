from accelerate import Accelerator
from accelerate.utils import gather_object

accelerator = Accelerator()

# each GPU creates a string
message = [ f"Hello this is GPU {accelerator.process_index}" ] 

# collect the messages from all GPUs
messages = gather_object(message)

# output the messages only on the main process with accelerator.print() 
accelerator.print(messages)