import inspect
import pathlib
import argparse
import threading
import itertools
import multiprocessing
import sys


def proc_run(sync_barrier: multiprocessing.Barrier
             , fin_event: multiprocessing.Event
             , args: argparse.Namespace):
  if args.e is None:
    experiment_names = list(_EXPERIMENTS.keys())
  else:
    experiment_names = args.e

  try:
    pid = sync_barrier.wait()
    device_string = str(args.cuda_device[pid])
    os.environ['CUDA_VISIBLE_DEVICES'] = device_string
    # Init torch in order to occupy GPU.
    torch.cuda.init()
    for experiment_name in experiment_names:
      sync_barrier.wait()
      out_log_path = os.path.join(args.dir, f'pid{os.getpid()}-{pid}_{experiment_name}.log')
      err_log_path = os.path.join(args.dir, f'pid{os.getpid()}-{pid}_{experiment_name}.err.log')
      sys.stdout = utils_io.LogFile(out_log_path, lazy_create=True)
      sys.stderr = utils_io.LogFile(err_log_path, lazy_create=True)
      print(f'CUDA_VISIBLE_DEVICES = {device_string}')
      experiment = _EXPERIMENTS[experiment_name]
      experiment(sync_barrier, pid, sync_barrier.parties, args)
  except threading.BrokenBarrierError:
    print('Aborted from outside!')
  finally:
    fin_event.set()


if __name__ == '__main__':
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Disable info logging of tensorflow
  main()