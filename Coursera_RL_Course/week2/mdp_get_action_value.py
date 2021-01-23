
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """
    Q = 0
    inter_states = mdp.get_next_states(state, action)
    for state_i, prob in inter_states.items():
        reward = mdp.get_reward(state, action, state_i)
        Q += prob*(reward+gamma*state_values[state_i])
    return Q
