from visualize.session import MonteCarloVizSession, SingleResultSession


def run_visualizer(*args, **kwargs):
    from visualize.app import run_visualizer as _run

    return _run(*args, **kwargs)


__all__ = [
    "MonteCarloVizSession",
    "SingleResultSession",
    "run_visualizer",
]
