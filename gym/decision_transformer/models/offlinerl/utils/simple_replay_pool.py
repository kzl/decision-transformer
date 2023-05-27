from collections import defaultdict

import numpy as np
from gym.spaces import Box, Dict, Discrete
import pdb

from .flexible_replay_pool import FlexibleReplayPool


def normalize_observation_fields(observation_space, name='observations'):
    if isinstance(observation_space, Dict):
        fields = [
            normalize_observation_fields(child_observation_space, name)
            for name, child_observation_space
            in observation_space.spaces.items()
        ]
        fields = {
            'observations.{}'.format(name): value
            for field in fields
            for name, value in field.items()
        }
    elif isinstance(observation_space, (Box, Discrete)):
        fields = {
            name: {
                'shape': observation_space.shape,
                'dtype': observation_space.dtype,
            }
        }
    else:
        raise NotImplementedError(
            "Observation space of type '{}' not supported."
            "".format(type(observation_space)))

    return fields


class SimpleReplayPool(FlexibleReplayPool):
    def __init__(self, observation_space, action_space, *args, **kwargs):
        self._observation_space = observation_space
        self._action_space = action_space

        active_size = sum(
            np.prod(observation_space.spaces[key].shape)
            for key in list(observation_space.spaces.keys()))

        active_observation_shape = (active_size, )

        fields = {
            'actions': {
                'shape': self._action_space.shape,
                'dtype': 'float32'
            },
            'last_actions': {
                'shape': self._action_space.shape,
                'dtype': 'float32'
            },
            'rewards': {
                'shape': (1, ),
                'dtype': 'float32'
            },
            # self.terminals[i] = a terminal was received at time i
            'terminals': {
                'shape': (1, ),
                'dtype': 'bool'
            },
            'valid': {
                'shape': (1, ),
                'dtype': 'float32'
            },
            'observations': {
                'shape': active_observation_shape,
                'dtype': 'float32'
            },
            'next_observations': {
                'shape': active_observation_shape,
                'dtype': 'float32'
            }
        }

        super(SimpleReplayPool, self).__init__(
            *args, fields_attrs=fields, **kwargs)

    def add_samples(self, samples):
        if not isinstance(self._observation_space, Dict):
            return super(SimpleReplayPool, self).add_samples(samples)

        dict_observations = defaultdict(list)
        return super(SimpleReplayPool, self).add_samples(samples)

    def batch_by_indices(self,
                         indices,
                         field_name_filter=None,
                         observation_keys=None):
        if not isinstance(self._observation_space, Dict):
            return super(SimpleReplayPool, self).batch_by_indices(
                indices, field_name_filter=field_name_filter)

        batch = {
            field_name: self.fields[field_name][indices]
            for field_name in self.field_names
        }

        if field_name_filter is not None:
            filtered_fields = self.filter_fields(
                batch.keys(), field_name_filter)
            batch = {
                field_name: batch[field_name]
                for field_name in filtered_fields
            }

        return batch

    def terminate_episode(self):
        pass

    def random_indices(self, batch_size):
        if self._size == 0: return np.arange(0, 0)
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size, field_name_filter=None, **kwargs):
        random_indices = self.random_indices(batch_size)
        return self.batch_by_indices(
            random_indices, field_name_filter=field_name_filter, **kwargs)


class SimpleReplayTrajPool(FlexibleReplayPool):
    def __init__(self, observation_space, action_space, max_traj_len, hidden_length, *args, **kwargs):
        self._observation_space = observation_space
        self._action_space = action_space
        self.max_traj_len = max_traj_len
        self.hidden_length = hidden_length

        fields = {
            'actions': {
                'shape': (self.max_traj_len, *self._action_space.shape),
                'dtype': 'float32'
            },
            'last_actions': {
                'shape': (self.max_traj_len, *self._action_space.shape),
                'dtype': 'float32'
            },
            'rewards': {
                'shape': (self.max_traj_len, *(1, )),
                'dtype': 'float32'
            },
            # self.terminals[i] = a terminal was received at time i
            'terminals': {
                'shape': (self.max_traj_len, *(1, )),
                'dtype': 'bool'
            },
            'valid': {
                'shape': (self.max_traj_len, *(1, )),
                'dtype': 'float32'
            },
            'observations': {
                'shape': (self.max_traj_len, *self._observation_space.shape),
                'dtype': 'float32'
            },
            'next_observations': {
                'shape': (self.max_traj_len, *self._observation_space.shape),
                'dtype': 'float32'
            },
            'policy_hidden': {
                'shape': (self.max_traj_len, self.hidden_length),
                'dtype': 'float32'
            },
            'value_hidden': {
                'shape': (self.max_traj_len, self.hidden_length),
                'dtype': 'float32'
            },
        }

        super(SimpleReplayTrajPool, self).__init__(
            *args, fields_attrs=fields, **kwargs)

    def add_samples(self, samples):
        if not isinstance(self._observation_space, Dict):
            return super(SimpleReplayTrajPool, self).add_samples(samples)

        dict_observations = defaultdict(list)
        return super(SimpleReplayTrajPool, self).add_samples(samples)

    def random_batch_for_initial(self, batch_size):
        # random_indices = self.random_indices(batch_size)
        valids = np.sum(self.fields['valid'], axis=1).squeeze(-1)[:self.size]
        first_ind = np.random.choice(np.arange(self.size), p=valids/np.sum(valids), size=(batch_size, ))
        second_ind = []
        for ind, item in enumerate(first_ind):
            second_ind.append(np.random.randint(valids[item]))
        indices = [(a, b) for a, b in zip(first_ind, second_ind)]
        return self.batch_by_double_index(
            indices)
    def batch_by_double_index(self, indices):
        batch = {

        }
        for field in self.field_names:
            shapes = self.fields[field].shape
            shapes = (len(indices), shapes[-1])
            data = np.zeros(shapes, dtype=np.float32)
            for ind, item in enumerate(indices):
                data[ind] = self.fields[field][item[0], item[1]]
            batch[field] = data
        return batch
    def batch_by_indices(self,
                         indices,
                         field_name_filter=None,
                         observation_keys=None):
        if not isinstance(self._observation_space, Dict):
            return super(SimpleReplayTrajPool, self).batch_by_indices(
                indices, field_name_filter=field_name_filter)

        batch = {
            field_name: self.fields[field_name][indices]
            for field_name in self.field_names
        }

        if field_name_filter is not None:
            filtered_fields = self.filter_fields(
                batch.keys(), field_name_filter)
            batch = {
                field_name: batch[field_name]
                for field_name in filtered_fields
            }

        return batch

    def terminate_episode(self):
        pass

    def random_indices(self, batch_size):
        if self._size == 0: return np.arange(0, 0)
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size, field_name_filter=None, **kwargs):
        random_indices = self.random_indices(batch_size)
        return self.batch_by_indices(
            random_indices, field_name_filter=field_name_filter, **kwargs)