class Trial():
    def __init__(self, trial_idx, poke_time, withdraw_time, odor_time, odor, position, in_seq, performance):
        self.trial_idx = trial_idx
        self.poke_time = poke_time
        self.withdraw_time = withdraw_time
        self.odor_time = odor_time
        self.odor = odor
        self.position = position
        self.in_seq = in_seq
        self.performance = performance
        # Different keys for self.spikes represent
        self.spikes = {}
        self.spikes['all'] = None
        self.spikes_sum_over_lfp = None
