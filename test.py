from dftracer.logger import dftracer
from dftracer.logger import dft_fn as Profile

log_inst = dftracer.initialize_log(
    logfile="test.pfw",
    data_dir=None,
    process_id=-1,
)

if log_inst is None:
    raise Exception("Failed to initialize dftracer")

dlp = Profile("test")

def gen():
    for i in range(10):
        yield i

for i in dlp.iter(gen()):
    if i == 8:
        break

log_inst.finalize()
