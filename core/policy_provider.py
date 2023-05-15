from typing import Type

from vmas.scenarios.transport import HeuristicPolicy as TransportPolicy, HeuristicPolicy
from vmas.scenarios.balance import HeuristicPolicy as BalancePolicy
from vmas.scenarios.wheel import HeuristicPolicy as WheelPolicy


class PolicyProvider:

    @staticmethod
    def get_policy_for(environment) -> Type[HeuristicPolicy]:
        if environment == "transport":
            return TransportPolicy
        if environment == "balance":
            return BalancePolicy
        if environment == "wheel":
            return WheelPolicy
        else:
            return None  # TODO: raise exception
