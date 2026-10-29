"""Apply the native runtime's configured CPU budget before importing Paddle."""
import os


def cpu_threads(config):
    options = config.get('pipeline_options', {})
    if not isinstance(options, dict):
        raise ValueError('pipeline_options must be a JSON object')
    count = options.get('cpu_threads', config.get('cpu_threads', 6))
    if type(count) is not int or not 1 <= count <= 256:
        raise ValueError('cpu_threads must be an integer from 1 to 256')
    return count


def configure_cpu_environment(config, environment=None):
    environment = os.environ if environment is None else environment
    count = cpu_threads(config)
    # The explicit project setting wins over a parent's stale OMP=1. This
    # mutates only the child environment, never the machine's settings.
    environment['OMP_NUM_THREADS'] = str(count)
    environment['MKL_NUM_THREADS'] = str(count)
    return count
