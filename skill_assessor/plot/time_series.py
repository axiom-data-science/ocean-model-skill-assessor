import matplotlib; matplotlib.use('agg')  # noqa

import matplotlib.pyplot as plt


def plot(reference, sample, title):
    _, ax = plt.subplots(1, 1, figsize=(16, 10))
    reference.plot(ax=ax)
    sample.plot(ax=ax)
    plt.title(title)
    plt.savefig('test.png')
