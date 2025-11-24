# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from omegaconf import OmegaConf

from nemo_skills.inference.model import BaseModel


class KnockoutTournamentManager:
    def __init__(
        self,
        llm: BaseModel,
        n_participants_per_tournament: int,
        prompt_config_path: str,
        llm_kwargs: dict,
        rng: np.random.RandomState,
    ):
        self.llm = llm
        self.n_participants_per_tournament = n_participants_per_tournament
        self.prompt_config_path = prompt_config_path
        self.llm_kwargs = llm_kwargs
        self.rng = rng

    def load_prompt_template(self, prompt_config_path: str) -> str:
        config = OmegaConf.load(prompt_config_path)
        assert getattr(config, "system", None) is None, "System prompt is not allowed"
        return config.user

    async def _llm_call(self, prompt: str, req_seed: int) -> Tuple[str, int]:
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm.generate_async(
            prompt=messages,
            **self.llm_kwargs,
            random_seed=req_seed,
        )
        full_response = response["generation"]
        output_tokens = response["num_generated_tokens"]
        llm_text = full_response.split("</think>")[-1].strip()
        return llm_text, output_tokens

    def format_participants(self, participants: List[Tuple[int, str]], common_context: Dict[str, Any]) -> str:
        """Format participants for the tournament prompt. To be implemented by subclasses."""
        raise NotImplementedError

    def extract_winner_from_result(
        self, tournament_result: str, participants: List[Tuple[int, str]], req_seed: int
    ) -> Tuple[int, List[int], bool]:
        """Extract winner from tournament result and return eliminated participant IDs and parsing success flag. To be implemented by subclasses."""
        raise NotImplementedError

    def validate_participant(self, participant: str) -> bool:
        """Validate if a participant is valid for the tournament. To be implemented by subclasses."""
        raise NotImplementedError

    async def run_single_game(
        self,
        participants: List[Tuple[int, str]],
        common_context: Dict[str, Any],
        req_seed: int,
    ) -> Tuple[int, List[int], Dict[str, Any]]:
        """Run a single game and return the winner, eliminated participant IDs, and tournament metadata."""
        tournament_prompt = self.load_prompt_template(self.prompt_config_path)
        tournament_prompt = tournament_prompt.format(**common_context, num_participants=len(participants))

        participants_text = self.format_participants(participants, common_context)
        tournament_prompt += "\n\n" + participants_text

        tournament_result, num_generated_tokens = await self._llm_call(tournament_prompt, req_seed)

        winner_idx, knocked_out_ids, parsing_success = self.extract_winner_from_result(
            tournament_result, participants, req_seed
        )

        game_log = "Winner Found" if parsing_success else "Winner randomly chosen due to parsing issue"

        tournament_log = {
            "participant_ids": [x[0] for x in participants],
            "num_generated_tokens": num_generated_tokens,
            "winner_idx": winner_idx,
            "knocked_out_ids": knocked_out_ids,
            "game_log": game_log,
            "parsing_success": parsing_success,
        }

        return winner_idx, knocked_out_ids, tournament_log

    async def run_tournament(
        self,
        participants: List[str],
        common_context: Dict[str, Any],
        winner_count: int,
    ) -> Tuple[List[int], List[int], List[Dict[str, Any]]]:
        """Run a game to select the best participants.

        Args:
            participants: List of participant strings
            common_context: Context to include in tournament prompts
            winner_count: Number of winners to select

        Returns:
            Tuple of (winner_indices, knocked_out_ids, tournament_logs)
        """
        assert winner_count > 0, "Number of winners to select must be greater than 0"
        knocked_out_ids = []
        all_tournament_logs = []

        indexed_participants = [(i, participant) for i, participant in enumerate(participants)]

        invalid_participants = []
        for i, participant in indexed_participants:
            if not self.validate_participant(participant):
                invalid_participants.append(i)

        if invalid_participants:
            knocked_out_ids.extend(invalid_participants)
            all_tournament_logs.append(
                {
                    "invalid_participants": invalid_participants,
                    "num_generated_tokens": 0,
                }
            )

        while len(knocked_out_ids) < len(participants) - winner_count:
            remaining_participants = [
                (i, participant) for i, participant in indexed_participants if i not in knocked_out_ids
            ]

            self.rng.shuffle(remaining_participants)

            if len(remaining_participants) < 2:
                break

            tournament_batches = []
            for i in range(0, len(remaining_participants), self.n_participants_per_tournament):
                batch = remaining_participants[i : i + self.n_participants_per_tournament]
                if len(batch) >= 2:  # Only run games with at least 2 participants
                    tournament_batches.append(batch)

            if not tournament_batches:
                break

            # Run all games in parallel
            tournament_tasks = [
                self.run_single_game(
                    batch,
                    common_context,
                    self.rng.randint(0, 100000),
                )
                for batch in tournament_batches
            ]

            tournament_results = await asyncio.gather(*tournament_tasks)

            for winner_idx, eliminated_ids, tournament_log in tournament_results:
                knocked_out_ids.extend(eliminated_ids)
                all_tournament_logs.append(tournament_log)

        # Keep only a subset of knocked_out_ids to ensure exactly winner_count winners in the final game
        total_participants = len(indexed_participants)
        participants_to_eliminate = total_participants - winner_count

        if len(knocked_out_ids) > participants_to_eliminate:
            knocked_out_ids = knocked_out_ids[:participants_to_eliminate]

        winner_indices = [i for i, _ in indexed_participants if i not in knocked_out_ids]

        return winner_indices, all_tournament_logs


class ProofKnockoutTournamentManager(KnockoutTournamentManager):
    """Tournament manager for proof selection."""

    PROOF_FORMAT = """[Proof {idx}]
{proof}"""

    def format_participants(self, participants: List[Tuple[int, str]], common_context: Dict[str, Any]) -> str:
        participants_text = ""
        for idx, (_, proof) in enumerate(participants, 1):
            participants_text += "\n\n" + self.PROOF_FORMAT.format(idx=idx, proof=proof)
        return participants_text

    def extract_winner_from_result(
        self, tournament_result: str, participants: List[Tuple[int, str]], req_seed: int
    ) -> Tuple[int, List[int], bool]:
        knocked_out_ids = []

        if "<best_solution>" in tournament_result and "</best_solution>" in tournament_result:
            try:
                winner_idx_str = tournament_result.split("<best_solution>")[-1].split("</best_solution>")[0].strip()
                winner_idx = int(winner_idx_str) - 1  # Convert to 0-based index
                winner_original_idx = participants[winner_idx][0]

                # Knock out all non-winners from this tournament
                for i, _ in participants:
                    if i != winner_original_idx:
                        knocked_out_ids.append(i)
                return winner_original_idx, knocked_out_ids, True
            except (ValueError, IndexError):
                pass

        # If we can't parse the result, randomly select a winner and eliminate the rest
        logging.warning(
            f"Proof tournament result parsing failed, randomly selecting a winner. Game result: {tournament_result}"
        )
        winner_original_idx = int(self.rng.choice([x[0] for x in participants]))
        for i, _ in participants:
            if i != winner_original_idx:
                knocked_out_ids.append(i)
        return winner_original_idx, knocked_out_ids, False

    def validate_participant(self, participant: str) -> bool:
        return isinstance(participant, str) and len(participant.strip()) > 0
