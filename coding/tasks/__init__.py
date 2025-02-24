import random
from .task import Task
from .fim import FillInMiddleTask
from .completion import CompletionTask
from .repo import RepoCompletionTask
from .repofile import RepoFileTask
from .debug import DebugTask
from .organic_convo import OrganicConvoTask

TASKS = {
    FillInMiddleTask.name: FillInMiddleTask,
    CompletionTask.name: CompletionTask,
    RepoCompletionTask.name: RepoCompletionTask,
    RepoFileTask.name: RepoFileTask,
    DebugTask.name: DebugTask,
}

from coding.repl import REPLClient
from coding.schemas import Context
from coding.helpers import Selector
from coding.datasets import DATASET_MANAGER
from coding.protocol import StreamCodeSynapse
from coding.datasets import GithubDataset, PipDataset

TASK_REGISTRY = {
    FillInMiddleTask.name: [GithubDataset.name],
    CompletionTask.name: [GithubDataset.name],
    RepoCompletionTask.name: [GithubDataset.name],
    RepoFileTask.name: [GithubDataset.name],
    DebugTask.name: [PipDataset.name]
}


def create_task(
    llm,
    task_name: str,
    selector: Selector = random.choice,
    repl: REPLClient = REPLClient()
) -> Task:
    """Create a task from the given task name and LLM pipeline.

    Args:
        llm (Pipeline): Pipeline to use for text generation
        task_name (str): Name of the task to create
        selector (Selector, optional): Selector function to choose a dataset. Defaults to random.choice.

    Raises:
        ValueError: If task_name is not a valid alias for a task, or if the task is not a subclass of Task
        ValueError: If no datasets are available for the given task
        ValueError: If the dataset for the given task is not found

    Returns:
        Task: Task instance
    """
    task = TASKS.get(task_name, None)
    if task is None or not issubclass(task, Task):
        raise ValueError(f"Task {task_name} not found")

    dataset_choices = TASK_REGISTRY.get(task_name, None)
    if len(dataset_choices) == 0:
        raise ValueError(f"No datasets available for task {task_name}")
    dataset_name = selector(dataset_choices)
    dataset = DATASET_MANAGER.datasets.get(dataset_name, None)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_name} not found")
    return task(
        llm=llm,
        context=dataset.next(**dict(task.dataset_options)),
        repl=repl
    )
    
def create_organic_task(
    llm,
    synapse: StreamCodeSynapse,
    repl: REPLClient = REPLClient(),
) -> Task:
    """Create a task from the given synapse and LLM pipeline."""
    
    return OrganicConvoTask(llm=llm, context=Context(messages=synapse.messages, files=synapse.files), repl=repl)
    
    