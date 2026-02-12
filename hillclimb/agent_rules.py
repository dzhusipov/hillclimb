"""Rule-based baseline agent for Hill Climb Racing."""

from __future__ import annotations

from hillclimb.controller import Action
from hillclimb.vision import GameState, VisionState


class RuleBasedAgent:
    """Simple heuristic agent that serves as a baseline.

    Rules:
    - If fuel < 10%: coast (nothing) to conserve
    - If tilt > 30  (nose up): gas to level out (in air, front-heavy helps)
    - If tilt < -30 (nose down): brake to level out
    - If going uphill (positive terrain slope): gas
    - Default: gas (keep moving forward)
    """

    def decide(self, state: VisionState, **kwargs) -> Action:
        if state.game_state != GameState.RACING:
            return Action.NOTHING

        # Very low fuel — coast to stretch remaining distance
        if state.fuel < 0.10:
            return Action.NOTHING

        # Severe tilt correction
        if state.tilt > 30:
            return Action.GAS
        if state.tilt < -30:
            return Action.BRAKE

        # Moderate tilt while airborne
        if state.airborne:
            if state.tilt > 15:
                return Action.GAS
            if state.tilt < -15:
                return Action.BRAKE
            return Action.NOTHING  # don't waste fuel in air

        # On ground: mostly gas, brake on steep downhill
        if state.terrain_slope < -20:
            return Action.BRAKE

        return Action.GAS
