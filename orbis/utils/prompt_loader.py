"""
Prompt loader for externalized prompt templates.
Supports YAML-based prompt templates with variable substitution.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

class PromptLoader:
    """Load and format prompt templates from YAML files"""

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._prompt_cache = {}

    def load(self, prompt_name: str, variables: dict[str, Any] = None) -> str:
        """
        Load a prompt template and format it with variables.

        Args:
            prompt_name: Name of the prompt file (without .yaml extension)
            variables: Dictionary of variables to substitute in the prompt

        Returns:
            Formatted prompt string
        """
        if variables is None:
            variables = {}

        # Load prompt template
        prompt_template = self._load_prompt_template(prompt_name)

        # Format developer and user prompts
        developer_prompt = prompt_template.get("developer_prompt", "")
        user_prompt = prompt_template.get("user_prompt", "")

        # Format with variables
        try:
            formatted_developer = developer_prompt.format(**variables) if developer_prompt else ""
            formatted_user = user_prompt.format(**variables) if user_prompt else ""

            # Combine developer and user prompts
            if formatted_developer and formatted_user:
                return f"{formatted_developer}\n\n{formatted_user}"
            else:
                return formatted_developer or formatted_user

        except KeyError as e:
            logger.error(f"Missing variable {e} for prompt {prompt_name}")
            raise ValueError(f"Missing variable {e} for prompt {prompt_name}") from e

    def get_developer_prompt(self, prompt_name: str, variables: dict[str, Any] = None) -> str:
        """Get only the developer prompt part"""
        if variables is None:
            variables = {}

        prompt_template = self._load_prompt_template(prompt_name)
        developer_prompt = prompt_template.get("developer_prompt", "")

        try:
            return developer_prompt.format(**variables) if developer_prompt else ""
        except KeyError as e:
            logger.error(f"Missing variable {e} for developer prompt {prompt_name}")
            raise ValueError(f"Missing variable {e} for developer prompt {prompt_name}") from e

    def get_user_prompt(self, prompt_name: str, variables: dict[str, Any] = None) -> str:
        """Get only the user prompt part"""
        if variables is None:
            variables = {}

        prompt_template = self._load_prompt_template(prompt_name)
        user_prompt = prompt_template.get("user_prompt", "")

        try:
            return user_prompt.format(**variables) if user_prompt else ""
        except KeyError as e:
            logger.error(f"Missing variable {e} for user prompt {prompt_name}")
            raise ValueError(f"Missing variable {e} for user prompt {prompt_name}") from e

    def _load_prompt_template(self, prompt_name: str) -> dict[str, str]:
        """Load prompt template from YAML file"""
        # Check cache first
        if prompt_name in self._prompt_cache:
            return self._prompt_cache[prompt_name]

        # Load from file
        prompt_file = self.prompts_dir / f"{prompt_name}.yaml"

        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        try:
            with open(prompt_file, encoding='utf-8') as f:
                template = yaml.safe_load(f)

            # Cache the template
            self._prompt_cache[prompt_name] = template
            return template

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {prompt_file}: {e}")
            raise ValueError(f"Error parsing prompt file {prompt_file}: {e}") from e
        except Exception as e:
            logger.error(f"Error loading prompt file {prompt_file}: {e}")
            raise

    def clear_cache(self):
        """Clear the prompt template cache"""
        self._prompt_cache.clear()
        logger.info("Prompt cache cleared")

    def list_available_prompts(self) -> list[str]:
        """List all available prompt templates"""
        if not self.prompts_dir.exists():
            return []

        return [f.stem for f in self.prompts_dir.glob("*.yaml")]
