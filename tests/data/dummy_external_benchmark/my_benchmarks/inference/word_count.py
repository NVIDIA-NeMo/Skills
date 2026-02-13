import hydra

from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig
from nemo_skills.utils import nested_dataclass


@nested_dataclass(kw_only=True)
class WordCountGenerationConfig(GenerationTaskConfig):
    # Add a custom flag that controls whether to do a verification step
    verify: bool = False


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_generation_config", node=WordCountGenerationConfig)


class WordCountGenerationTask(GenerationTask):
    """Generation task with an optional verification step."""

    async def process_single_datapoint(self, data_point, all_data):
        # Step 1: normal generation
        result = await super().process_single_datapoint(data_point, all_data)

        if not self.cfg.verify:
            return result

        # Step 2: ask the model to verify its own answer
        verify_prompt = (
            f"You previously answered the following question:\n\n"
            f"{data_point['problem']}\n\n"
            f"Your answer was:\n{result['generation']}\n\n"
            f"Please verify this is correct. "
            f"If it is, repeat the same answer inside \\boxed{{}}. "
            f"If not, provide the corrected answer inside \\boxed{{}}."
        )
        new_data_point = [{"role": "user", "content": verify_prompt}]
        # We use prompt_format=openai as we already prepared the full message
        verify_result = await super().process_single_datapoint(
            new_data_point,
            all_data,
            prompt_format="openai",
        )
        # Replace generation with the verified answer
        result["generation"] = verify_result["generation"]
        return result


GENERATION_TASK_CLASS = WordCountGenerationTask


@hydra.main(version_base=None, config_name="base_generation_config")
def generate(cfg: WordCountGenerationConfig):
    cfg = WordCountGenerationConfig(_init_nested=True, **cfg)
    task = WordCountGenerationTask(cfg)
    task.generate()


if __name__ == "__main__":
    generate()
