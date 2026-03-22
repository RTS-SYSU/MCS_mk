class Job:
    """
       Represents a job instance at runtime.
    """

    def __init__(self, task_id: int, job_id: int, arrival_time: float, deadline: float,
                 wcet: float, actual_exec_time: float, priority: int, is_optional: bool, criticality: str):
        self.task_id = task_id
        self.job_id = job_id
        self.arrival_time = arrival_time
        self.absolute_deadline = deadline
        self.initial_wcet = wcet
        # Physical remaining time based on ACET.
        # This determines when the job actually finishes and leaves the CPU.
        self.remaining_time = actual_exec_time
        # # Logical remaining budget based on WCET.
        # Used for pessimistic Slack Calculation (Safety).
        self.remaining_budget = wcet
        self.priority = priority
        self.is_optional = is_optional
        self.criticality = criticality
        self.start_time = -1.0
        self.finish_time = -1.0

    def __repr__(self):
        return (f"Job(T{self.task_id}-{self.job_id}|{'OPT' if self.is_optional else 'MAND'}"
                f"|RemT:{self.remaining_time:.2f}|Budg:{self.remaining_budget:.2f})")
