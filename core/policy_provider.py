from typing import Type

from vmas.scenarios.transport import HeuristicPolicy as TransportPolicy, HeuristicPolicy
from vmas.scenarios.wheel import HeuristicPolicy as WheelPolicy


class PolicyProvider:

    @staticmethod
    def get_policy_for(environment) -> Type[HeuristicPolicy]:
        if environment == "transport" or environment == "reverse_transport":
            return TransportPolicy
        elif environment == "wheel":
            return WheelPolicy
        else:
            return None  # TODO: raise exception
