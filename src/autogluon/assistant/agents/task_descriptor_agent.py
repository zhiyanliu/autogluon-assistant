import logging

from ..prompts import TaskDescriptorPrompt
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


class TaskDescriptorAgent(BaseAgent):
    """
    Generate task description based on data prompt, description files, and analysis.

    Agent Input:
    - data_prompt: Text string containing data prompt
    - description_files: List of description filenames
    - description_analysis: Analysis from previous step

    Agent Output:
    - Generated task description string
    """

    def __init__(self, config, manager, llm_config, prompt_template):
        super().__init__(config=config, manager=manager)

        self.task_descriptor_llm_config = llm_config
        self.task_descriptor_prompt_template = prompt_template

        self.task_descriptor_prompt = TaskDescriptorPrompt(
            llm_config=self.task_descriptor_llm_config,
            manager=self.manager,
            template=self.task_descriptor_prompt_template,
        )

        if self.task_descriptor_llm_config.multi_turn:
            self.task_descriptor_llm = init_llm(
                llm_config=self.task_descriptor_llm_config,
                agent_name="task_descriptor",
                multi_turn=self.task_descriptor_llm_config.multi_turn,
            )

    def __call__(
        self,
    ):
        """
        Generate task description based on provided data and analysis.

        Returns:
            str: Generated task description
        """
        self.manager.log_agent_start(
            "TaskDescriptorAgent: generating a concise task description from source materials."
        )

        # Attach description file directly if within certain length
        description_files_contents = self.task_descriptor_prompt.get_description_files_contents(to_show=True)

        # Honor the agent-level config knob max_description_files_length_to_show: when the
        # combined description files are short enough to fit, skip the LLM summarization and use
        # the file contents verbatim as the task description. This both saves an LLM round-trip
        # and preserves the source text faithfully (no summarization noise).
        to_show_max = getattr(self.task_descriptor_llm_config, "max_description_files_length_to_show", 1024)
        if description_files_contents and len(description_files_contents) <= to_show_max:
            task_description = description_files_contents
            # Mirror the side effects of TaskDescriptorPrompt.parse() so downstream tooling that
            # reads task_description.txt still works on this fast path.
            self.manager.save_and_log_states(
                content=task_description,
                save_name="task_description.txt",
                per_iteration=False,
                add_uuid=False,
            )
            self.manager.log_agent_end(
                "TaskDescriptorAgent: description files used verbatim (under "
                f"max_description_files_length_to_show={to_show_max} chars), LLM summarization skipped."
            )
            return task_description

        task_description = description_files_contents

        # Otherwise generate condensed task description
        prompt = self.task_descriptor_prompt.build()

        if not self.task_descriptor_llm_config.multi_turn:
            self.task_descriptor_llm = init_llm(
                llm_config=self.task_descriptor_llm_config,
                agent_name="task_descriptor",
                multi_turn=self.task_descriptor_llm_config.multi_turn,
            )

        response = self.task_descriptor_llm.assistant_chat(prompt)

        task_description = self.task_descriptor_prompt.parse(response)

        self.manager.log_agent_end("TaskDescriptorAgent: task description generated.")

        return task_description
