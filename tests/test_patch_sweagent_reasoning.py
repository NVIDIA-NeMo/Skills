from nemo_skills.inference.eval.patch_sweagent_reasoning import patch_agents_py, patch_tools_py


def test_patch_agents_caps_problem_statement_environment_mirror(tmp_path):
    agents_path = tmp_path / "sweagent" / "agent" / "agents.py"
    agents_path.parent.mkdir(parents=True)
    agents_path.write_text(
        "step.reasoning_content = output.get('reasoning_content')\n"
        "reasoning_history = step.reasoning_content.rstrip('\\n')\n"
        "        self._env.set_env_variables({\"PROBLEM_STATEMENT\": "
        "self._problem_statement.get_problem_statement_for_env()})\n"
    )

    patch_agents_py(tmp_path)
    patched = agents_path.read_text()

    assert "if len(problem_statement_env) > 4000:" in patched
    assert "problem_statement_env[:4000]" in patched
    patch_agents_py(tmp_path)


def test_patch_tools_combines_reset_writes_into_one_rpc(tmp_path):
    tools_path = tmp_path / "sweagent" / "tools" / "tools.py"
    tools_path.parent.mkdir(parents=True)
    tools_path.write_text(
        "import re\n"
        "\n"
        "class Tools:\n"
        "    def reset(self, env: SWEEnv) -> None:\n"
        "        self.logger.info(\"Resetting tools\")\n"
        "        env_variables = self.config.env_variables.copy() | {\n"
        "            var: os.getenv(var) for var in self.config.propagate_env_variables\n"
        "        }\n"
        "        env.set_env_variables(env_variables)\n"
        "        env.write_file(\"/root/.swe-agent-env\", json.dumps(self.config.registry_variables))\n"
        "        env.write_file(\"/root/state.json\", \"{}\")\n"
        "        env.communicate(\" && \".join(self._reset_commands), check=\"raise\", "
        "timeout=self.config.install_timeout)\n"
    )

    patch_tools_py(tmp_path)
    patched = tools_path.read_text()

    assert "import shlex" in patched
    assert "reset_commands = [" in patched
    assert "Tools reset complete" in patched
    assert "env.set_env_variables" not in patched
    assert "env.write_file" not in patched
    patch_tools_py(tmp_path)
