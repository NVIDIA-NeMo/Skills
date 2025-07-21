#!/usr/bin/env python3
"""
Thread-Safe Interactive Lean 4 Development Agent

This recreates the VS Code Lean 4 extension experience programmatically with thread safety:
- Real-time compiler feedback and messages
- Position-aware editing with goal state tracking
- Targeted updates with immediate validation
- Interactive development workflow for LLM agents
- Thread-safe concurrent usage

Mimics how mathematicians work with Lean 4 while ensuring multiple agents don't interfere.
"""

import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, NamedTuple

from .prover import LeanProver


class Position(NamedTuple):
    """Position in the Lean file (line, column)."""
    line: int
    column: int


@dataclass
class LeanMessage:
    """A compiler message (error, warning, info) at a specific position."""
    severity: str  # 'error', 'warning', 'info'
    message: str
    start_pos: Position
    end_pos: Position

    def __str__(self) -> str:
        return f"[{self.severity}] {self.message} at line {self.start_pos.line + 1}"


@dataclass
class ProofGoal:
    """A proof goal at a specific position."""
    goal_text: str
    position: Position
    proof_state_id: Optional[int] = None

    def __str__(self) -> str:
        return f"Goal at line {self.position.line + 1}: {self.goal_text[:100]}..."


@dataclass
class EditableClause:
    """A clause/section of code that can be edited."""
    clause_id: str
    start_pos: Position
    end_pos: Position
    content: str
    clause_type: str  # 'tactic', 'have', 'main_goal', 'sorry', etc.

    def __str__(self) -> str:
        return f"{self.clause_type} [{self.clause_id}]: {self.content[:50]}..."


class InteractiveLeanAgent:
    """
    Thread-safe Interactive Lean 4 development agent that mimics VS Code extension.

    Features:
    - Real-time compilation and feedback
    - Position-aware editing
    - Goal state tracking
    - Targeted clause updates
    - Incremental development workflow
    - Thread-safe concurrent usage
    """

    def __init__(self, mathlib_enabled: bool = True):
        """Initialize thread-safe interactive agent."""
        self._mathlib_enabled = mathlib_enabled
        self._lock = threading.RLock()
        self._thread_local = threading.local()

        # Instance ID for debugging
        self._instance_id = str(uuid.uuid4())[:8]

    def _get_agent_state(self):
        """Get or create thread-local agent state."""
        if not hasattr(self._thread_local, 'prover'):
            # Each thread gets its own state
            self._thread_local.prover = LeanProver(mathlib_enabled=self._mathlib_enabled)
            self._thread_local.current_code = ""
            self._thread_local.current_messages = []
            self._thread_local.current_goals = []
            self._thread_local.editable_clauses = {}
            self._thread_local.compilation_id = 0
            self._thread_local.thread_id = threading.get_ident()
            self._thread_local.session_id = str(uuid.uuid4())[:8]

        return self._thread_local

    @property
    def prover(self) -> LeanProver:
        """Get thread-local prover."""
        state = self._get_agent_state()
        return state.prover

    @property
    def current_code(self) -> str:
        """Get current code for this thread."""
        state = self._get_agent_state()
        return state.current_code

    @property
    def current_messages(self) -> List[LeanMessage]:
        """Get current messages for this thread."""
        state = self._get_agent_state()
        return state.current_messages

    @property
    def current_goals(self) -> List[ProofGoal]:
        """Get current goals for this thread."""
        state = self._get_agent_state()
        return state.current_goals

    @property
    def editable_clauses(self) -> Dict[str, EditableClause]:
        """Get editable clauses for this thread."""
        state = self._get_agent_state()
        return state.editable_clauses

    @property
    def compilation_id(self) -> int:
        """Get compilation ID for this thread."""
        state = self._get_agent_state()
        return state.compilation_id

    def load_theorem(self, theorem_code: str) -> Dict[str, Any]:
        """
        Thread-safe theorem loading and compilation.
        Returns compilation results with messages and goals.
        """
        state = self._get_agent_state()
        state.current_code = theorem_code

        # Add thread-specific markers to avoid naming conflicts
        thread_id = threading.get_ident()
        session_id = state.session_id

        # Modify theorem code to be thread-unique
        modified_code = self._make_thread_unique_code(theorem_code, thread_id, session_id)

        # Compile and get feedback
        result = self._compile_and_analyze(modified_code)

        # Add thread info to result
        result['thread_info'] = {
            'thread_id': thread_id,
            'session_id': session_id,
            'instance_id': self._instance_id
        }

        return result

    def _make_thread_unique_code(self, code: str, thread_id: int, session_id: str) -> str:
        """Make theorem code unique per thread to avoid conflicts."""
        # Find theorem names and make them unique
        def replace_theorem_name(match):
            original_name = match.group(1)
            unique_name = f"{original_name}_t{thread_id % 10000}_s{session_id}"
            return f"theorem {unique_name}"

        # Replace theorem declarations
        unique_code = re.sub(r'theorem\s+(\w+)', replace_theorem_name, code)
        return unique_code

    def _compile_and_analyze(self, unique_code: str) -> Dict[str, Any]:
        """Compile current code and analyze results."""
        state = self._get_agent_state()

        # Increment compilation ID for each compilation
        state.compilation_id += 1

        # Compile with lean-interact
        result = state.prover.run_command(unique_code)

        # Parse messages and goals
        self._parse_messages(result)
        self._parse_goals(result)
        self._identify_editable_clauses()

        # Better success detection - check if we have actual errors vs just warnings
        has_errors = any(msg.severity == 'error' for msg in state.current_messages)
        compilation_success = not has_errors  # Success means no errors (warnings are OK)

        return {
            "success": compilation_success,
            "has_errors": has_errors,
            "has_warnings": any(msg.severity == 'warning' for msg in state.current_messages),
            "has_sorry": result.has_sorry,
            "messages": state.current_messages,
            "goals": state.current_goals,
            "editable_clauses": list(state.editable_clauses.keys()),
            "proof_state": result.proof_state,
            "compilation_id": state.compilation_id,
            "raw_response": result.response
        }

    def _parse_messages(self, result):
        """Parse compiler messages from lean-interact result."""
        state = self._get_agent_state()
        state.current_messages = []

        if hasattr(result, 'response') and result.response:
            response = result.response

            # Parse different types of messages with better detection
            if '[error]' in response:
                # Extract error message
                error_match = re.search(r'\[error\]\s*(.*?)(?=\n\n|\n\[|$)', response, re.DOTALL)
                if error_match:
                    error_text = error_match.group(1).strip()
                    state.current_messages.append(LeanMessage(
                        severity='error',
                        message=error_text,
                        start_pos=Position(0, 0),
                        end_pos=Position(0, 0)
                    ))

            if '[warning]' in response:
                # Extract warning message
                warning_match = re.search(r'\[warning\]\s*(.*?)(?=\n\n|\n\[|$)', response, re.DOTALL)
                if warning_match:
                    warning_text = warning_match.group(1).strip()
                    state.current_messages.append(LeanMessage(
                        severity='warning',
                        message=warning_text,
                        start_pos=Position(0, 0),
                        end_pos=Position(0, 0)
                    ))

            if '[info]' in response:
                # Extract info message
                info_match = re.search(r'\[info\]\s*(.*?)(?=\n\n|\n\[|$)', response, re.DOTALL)
                if info_match:
                    info_text = info_match.group(1).strip()
                    state.current_messages.append(LeanMessage(
                        severity='info',
                        message=info_text,
                        start_pos=Position(0, 0),
                        end_pos=Position(0, 0)
                    ))

        # If response is just "Success", add success message
        if hasattr(result, 'response') and result.response and result.response.strip() == "Success":
            state.current_messages.append(LeanMessage(
                severity='info',
                message='Compilation successful',
                start_pos=Position(0, 0),
                end_pos=Position(0, 0)
            ))

        # If no messages but the command succeeded, add a success note
        elif not state.current_messages and hasattr(result, 'response'):
            state.current_messages.append(LeanMessage(
                severity='info',
                message='Compilation completed',
                start_pos=Position(0, 0),
                end_pos=Position(0, 0)
            ))

    def _parse_goals(self, result):
        """Parse proof goals from lean-interact result."""
        state = self._get_agent_state()
        state.current_goals = []

        if hasattr(result, 'response') and result.response:
            # Look for goal patterns in response
            lines = state.current_code.split('\n')
            for i, line in enumerate(lines):
                if 'sorry' in line:
                    state.current_goals.append(ProofGoal(
                        goal_text=f"Goal at sorry on line {i+1}",
                        position=Position(i, line.find('sorry')),
                        proof_state_id=result.proof_state
                    ))

    def _identify_editable_clauses(self):
        """Identify editable clauses in the current code."""
        state = self._get_agent_state()
        state.editable_clauses = {}
        lines = state.current_code.split('\n')

        clause_id = 0
        main_proof_started = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith('--'):
                continue

            # Check if we're in the main theorem declaration
            if stripped.startswith('theorem'):
                # Look for ":= by" pattern to identify main proof start
                if ':= by' in stripped:
                    # Extract the part after ":= by" on same line
                    by_part = stripped.split(':= by', 1)[1].strip()
                    if by_part:
                        # Main proof content on same line as theorem declaration
                        state.editable_clauses[f"main_proof_{clause_id}"] = EditableClause(
                            clause_id=f"main_proof_{clause_id}",
                            start_pos=Position(i, stripped.find(':= by') + 5),
                            end_pos=Position(i, len(line)),
                            content=by_part,
                            clause_type='main_proof'
                        )
                        clause_id += 1
                    main_proof_started = True
                continue

            # Skip non-proof lines after theorem
            if main_proof_started:
                # Identify different types of editable clauses

                if stripped.startswith('have '):
                    # Have statement - extract variable name and proof
                    have_match = re.match(r'have\s+(\w+)\s*:.*?:=\s*by\s*(.*)', stripped)
                    if have_match:
                        var_name = have_match.group(1)
                        proof_content = have_match.group(2).strip()

                        state.editable_clauses[f"have_{var_name}"] = EditableClause(
                            clause_id=f"have_{var_name}",
                            start_pos=Position(i, line.find('by') + 3),
                            end_pos=Position(i, len(line)),
                            content=proof_content if proof_content else 'sorry',
                            clause_type='have'
                        )
                        clause_id += 1

                elif 'sorry' in stripped:
                    # Sorry clause - can be standalone or inline
                    sorry_pos = line.find('sorry')
                    state.editable_clauses[f"sorry_{clause_id}"] = EditableClause(
                        clause_id=f"sorry_{clause_id}",
                        start_pos=Position(i, sorry_pos),
                        end_pos=Position(i, sorry_pos + 5),
                        content='sorry',
                        clause_type='sorry'
                    )
                    clause_id += 1

                elif stripped.startswith('by '):
                    # Standalone "by" with tactics
                    tactic_content = stripped[3:].strip()  # Remove "by "
                    state.editable_clauses[f"tactic_block_{clause_id}"] = EditableClause(
                        clause_id=f"tactic_block_{clause_id}",
                        start_pos=Position(i, line.find('by') + 3),
                        end_pos=Position(i, len(line)),
                        content=tactic_content,
                        clause_type='tactic_block'
                    )
                    clause_id += 1

                elif any(stripped.startswith(tactic) for tactic in [
                    'intro ', 'exact ', 'apply ', 'rw ', 'simp', 'trivial',
                    'rfl', 'constructor', 'cases ', 'induction ', 'unfold ',
                    'left', 'right', 'split', 'exfalso', 'contradiction'
                ]):
                    # Individual tactic lines
                    state.editable_clauses[f"tactic_{clause_id}"] = EditableClause(
                        clause_id=f"tactic_{clause_id}",
                        start_pos=Position(i, 0),
                        end_pos=Position(i, len(line)),
                        content=stripped,
                        clause_type='tactic'
                    )
                    clause_id += 1

                elif stripped and not stripped.startswith('theorem'):
                    # Any other non-empty line in proof context - make it editable
                    state.editable_clauses[f"proof_line_{clause_id}"] = EditableClause(
                        clause_id=f"proof_line_{clause_id}",
                        start_pos=Position(i, 0),
                        end_pos=Position(i, len(line)),
                        content=stripped,
                        clause_type='proof_line'
                    )
                    clause_id += 1

    def get_goal_at_position(self, line: int, column: int) -> Optional[ProofGoal]:
        """Get the proof goal at a specific position (like VS Code hover)."""
        for goal in self.current_goals:
            if goal.position.line == line:
                return goal
        return None

    def get_messages_at_position(self, line: int, column: int) -> List[LeanMessage]:
        """Get compiler messages at a specific position."""
        messages = []
        for msg in self.current_messages:
            if msg.start_pos.line <= line <= msg.end_pos.line:
                messages.append(msg)
        return messages

    def edit_clause(self, clause_id: str, new_content: str) -> Dict[str, Any]:
        """
        Edit a specific clause and get immediate feedback.
        This is the core interactive editing function.
        """
        state = self._get_agent_state()

        if clause_id not in state.editable_clauses:
            result = {"error": f"Clause '{clause_id}' not found"}
            result['thread_info'] = {
                'thread_id': threading.get_ident(),
                'session_id': getattr(state, 'session_id', 'unknown'),
                'instance_id': self._instance_id
            }
            return result

        clause = state.editable_clauses[clause_id]

        # Apply the edit
        lines = state.current_code.split('\n')
        line_idx = clause.start_pos.line
        line = lines[line_idx]

        # Replace the clause content based on clause type
        if clause.clause_type == 'sorry':
            # Replace sorry with new content
            new_line = line.replace('sorry', new_content)

        elif clause.clause_type == 'have':
            # Replace the part after 'by' in have statement
            by_pos = line.find('by')
            if by_pos != -1:
                new_line = line[:by_pos + 2] + " " + new_content
            else:
                new_line = line

        elif clause.clause_type == 'main_proof':
            # Replace the part after ':= by' in theorem declaration
            by_pos = line.find(':= by')
            if by_pos != -1:
                new_line = line[:by_pos + 5] + " " + new_content
            else:
                new_line = line

        elif clause.clause_type == 'tactic_block':
            # Replace content after 'by '
            by_pos = line.find('by')
            if by_pos != -1:
                new_line = line[:by_pos + 2] + " " + new_content
            else:
                new_line = line

        elif clause.clause_type in ['tactic', 'proof_line']:
            # Replace entire line content, preserving indentation
            indent = len(line) - len(line.lstrip())
            new_line = ' ' * indent + new_content

        else:
            # Default: replace from start position to end
            start_col = clause.start_pos.column
            new_line = line[:start_col] + new_content

        lines[line_idx] = new_line
        state.current_code = '\n'.join(lines)

        # Recompile and get feedback
        modified_code = self._make_thread_unique_code(
            state.current_code,
            state.thread_id,
            state.session_id
        )
        compile_result = self._compile_and_analyze(modified_code)

        result = {
            "clause_id": clause_id,
            "clause_type": clause.clause_type,
            "old_content": clause.content,
            "new_content": new_content,
            "compilation_result": compile_result,
            "updated_code": state.current_code,
            "edit_successful": True
        }

        # Add thread info
        result['thread_info'] = {
            'thread_id': threading.get_ident(),
            'session_id': getattr(state, 'session_id', 'unknown'),
            'instance_id': self._instance_id
        }

        return result

    def add_proof_structure(self, structure_lines: List[str]) -> Dict[str, Any]:
        """
        Thread-safe helper method to add proof structure (like have clauses) to a simple theorem.

        Args:
            structure_lines: List of proof lines to add (e.g., ["have h1 : ... := by sorry", "intro x", "exact h1 x"])

        Returns:
            Edit result with updated editable clauses
        """
        # Join the structure lines with proper indentation
        indented_lines = []
        for line in structure_lines:
            if line.strip():  # Skip empty lines
                # Add consistent indentation
                if not line.startswith('  '):
                    indented_lines.append('  ' + line.strip())
                else:
                    indented_lines.append(line.rstrip())

        # Create multi-line proof structure
        new_structure = '\n'.join(indented_lines)

        # Find the main proof clause to edit
        state = self._get_agent_state()
        main_proof_clauses = [cid for cid in state.editable_clauses.keys()
                             if cid.startswith('main_proof_') or cid.startswith('sorry_')]

        if not main_proof_clauses:
            result = {"error": "No main proof clause found to add structure to"}
            result['thread_info'] = {
                'thread_id': threading.get_ident(),
                'session_id': getattr(state, 'session_id', 'unknown'),
                'instance_id': self._instance_id
            }
            return result

        # Edit the first available main proof clause
        clause_id = main_proof_clauses[0]
        return self.edit_clause(clause_id, new_structure)

    def get_proof_structure_suggestions(self) -> List[str]:
        """
        Suggest common proof structure patterns based on the theorem type.
        """
        suggestions = [
            "# Common patterns you can add:",
            "",
            "# For implications (P → Q):",
            ["have h1 : P := by sorry", "have h2 : P → Q := by sorry", "exact h2 h1"],
            "",
            "# For conjunctions (P ∧ Q):",
            ["have h1 : P := by sorry", "have h2 : Q := by sorry", "exact ⟨h1, h2⟩"],
            "",
            "# For complex proofs:",
            ["have lemma1 : ... := by sorry", "have lemma2 : ... := by sorry", "-- main proof steps", "exact lemma1"],
        ]
        return suggestions

    def get_interactive_panel(self) -> Dict[str, Any]:
        """
        Get the current state of the 'interactive panel' - like VS Code's side panel.
        """
        state = self._get_agent_state()

        panel = {
            "current_code": state.current_code,
            "messages": [str(msg) for msg in state.current_messages],
            "goals": [str(goal) for goal in state.current_goals],
            "editable_clauses": {
                cid: f"{clause.clause_type}: {clause.content}"
                for cid, clause in state.editable_clauses.items()
            },
            "compilation_id": state.compilation_id
        }

        # Add thread-specific info
        panel['thread_info'] = {
            'thread_id': threading.get_ident(),
            'session_id': getattr(state, 'session_id', 'unknown'),
            'instance_id': self._instance_id
        }

        return panel

    def suggest_next_actions(self) -> List[str]:
        """
        Thread-safe action suggestions based on current state.
        """
        state = self._get_agent_state()
        suggestions = []

        # If there are errors, suggest fixing them
        error_messages = [msg for msg in state.current_messages if msg.severity == 'error']
        if error_messages:
            suggestions.append("Fix compilation errors first")

        # If there are sorries, suggest working on them
        sorry_clauses = [cid for cid, clause in state.editable_clauses.items()
                        if clause.clause_type == 'sorry']
        if sorry_clauses:
            suggestions.append(f"Work on sorry clauses: {', '.join(sorry_clauses)}")

        # If there are warnings, suggest addressing them
        warning_messages = [msg for msg in state.current_messages if msg.severity == 'warning']
        if warning_messages:
            suggestions.append("Address compiler warnings")

        if not suggestions:
            suggestions.append("Proof looks complete!")

        return suggestions

    def get_thread_info(self) -> Dict[str, Any]:
        """Get debugging info about current thread's agent state."""
        if not hasattr(self._thread_local, 'prover'):
            return {
                'instance_id': self._instance_id,
                'thread_id': threading.get_ident(),
                'status': 'no_agent_initialized'
            }

        state = self._get_agent_state()
        return {
            'instance_id': self._instance_id,
            'thread_id': threading.get_ident(),
            'session_id': getattr(state, 'session_id', 'unknown'),
            'compilation_id': state.compilation_id,
            'current_code_length': len(state.current_code),
            'active_messages': len(state.current_messages),
            'active_goals': len(state.current_goals),
            'editable_clauses': len(state.editable_clauses)
        }


# Demo function remains for backward compatibility
def demo_interactive_agent():
    """Demonstrate the interactive Lean 4 development experience."""
    print("=" * 80)
    print("INTERACTIVE LEAN 4 DEVELOPMENT AGENT")
    print("=" * 80)

    agent = InteractiveLeanAgent(mathlib_enabled=True)

    print("\n🎯 LOADING INITIAL THEOREM:")
    print("-" * 60)

    # Load a theorem with multiple parts to work on
    theorem_code = """theorem interactive_demo (P Q R : Prop) : (P ∧ Q) ∧ R → P ∧ (Q ∧ R) := by
  have h1 : (P ∧ Q) ∧ R → P ∧ Q := by sorry
  have h2 : (P ∧ Q) ∧ R → R := by sorry
  have h3 : P ∧ Q → P := by sorry
  have h4 : P ∧ Q → Q := by sorry
  intro h
  exact ⟨h3 (h1 h), ⟨h4 (h1 h), h2 h⟩⟩"""

    result = agent.load_theorem(theorem_code)

    print("INITIAL CODE:")
    print(theorem_code)
    print()

    # Show interactive panel
    panel = agent.get_interactive_panel()
    print("📊 INTERACTIVE PANEL:")
    print(f"  Messages: {len(panel['messages'])}")
    for msg in panel['messages']:
        print(f"    {msg}")
    print(f"  Goals: {len(panel['goals'])}")
    for goal in panel['goals']:
        print(f"    {goal}")
    print(f"  Editable clauses: {len(panel['editable_clauses'])}")
    for cid, desc in panel['editable_clauses'].items():
        print(f"    {cid}: {desc}")

    print("\n🎯 INTERACTIVE EDITING SESSION:")
    print("-" * 60)

    # Simulate LLM working on the proof interactively
    print("🤖 LLM Agent starts working...")

    # Edit h1
    print("\n1. Working on h1...")
    edit_result = agent.edit_clause("have_h1", "intro h; exact h.left")
    print(f"   Edit result: {edit_result['compilation_result']['success']}")
    print(f"   Messages: {len(edit_result['compilation_result']['messages'])}")

    # Edit h2
    print("\n2. Working on h2...")
    edit_result = agent.edit_clause("have_h2", "intro h; exact h.right")
    print(f"   Edit result: {edit_result['compilation_result']['success']}")
    print(f"   Messages: {len(edit_result['compilation_result']['messages'])}")

    # Edit h3
    print("\n3. Working on h3...")
    edit_result = agent.edit_clause("have_h3", "intro h; exact h.left")
    print(f"   Edit result: {edit_result['compilation_result']['success']}")
    print(f"   Messages: {len(edit_result['compilation_result']['messages'])}")

    # Edit h4
    print("\n4. Working on h4...")
    edit_result = agent.edit_clause("have_h4", "intro h; exact h.right")
    print(f"   Edit result: {edit_result['compilation_result']['success']}")
    print(f"   Messages: {len(edit_result['compilation_result']['messages'])}")

    print("\n📊 FINAL INTERACTIVE PANEL:")
    print("-" * 60)

    final_panel = agent.get_interactive_panel()
    print("FINAL CODE:")
    print(final_panel['current_code'])
    print()

    print("FINAL MESSAGES:")
    for msg in final_panel['messages']:
        print(f"  {msg}")

    print("\n🎯 NEXT ACTION SUGGESTIONS:")
    suggestions = agent.suggest_next_actions()
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")

    print("\n🎉 INTERACTIVE DEVELOPMENT COMPLETE!")
    print("This mimics the VS Code Lean 4 extension experience!")


if __name__ == "__main__":
    demo_interactive_agent()

    print("\n" + "=" * 80)
    print("🎯 INTERACTIVE LEAN 4 AGENT SUMMARY:")
    print("=" * 80)
    print()
    print("✅ MIMICS HUMAN DEVELOPMENT WORKFLOW:")
    print("• Real-time compilation and feedback")
    print("• Position-aware editing with goal tracking")
    print("• Targeted clause updates")
    print("• Interactive panel showing messages/goals")
    print("• Incremental development with immediate validation")
    print()
    print("✅ KEY FEATURES:")
    print("• load_theorem(): Load and analyze initial code")
    print("• edit_clause(): Make targeted edits with feedback")
    print("• get_interactive_panel(): Get current state")
    print("• get_goal_at_position(): Get goals at cursor position")
    print("• suggest_next_actions(): AI-driven development suggestions")
    print()
    print("✅ PERFECT FOR LLM AGENTS:")
    print("• Mirrors how mathematicians work with Lean 4")
    print("• Provides immediate feedback after each edit")
    print("• Enables iterative proof development")
    print("• Supports complex proof construction workflows")
    print("• Thread-safe for concurrent usage")
    print()
    print("🎉 Ready for LLM-driven interactive theorem proving!")
    print("=" * 80)
