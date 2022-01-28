"""
Copyright (C) 2020

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import inspect
import functools
import types

import numpy as np
import torch
import glom

import lamarckian


def immediate(get_feature, coding='reward_blob'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(mdp):
        class MDP(mdp):
            class Controller(mdp.Controller):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    assert not hasattr(self, PATH_FUNC)

                def get_state(self):
                    state = super().get_state()
                    if not hasattr(self, PATH_FUNC):
                        setattr(self, PATH_FUNC, get_feature(self))
                    return state

                def get_reward(self):
                    feature = getattr(self, PATH_FUNC)
                    feature_ = get_feature(self)
                    setattr(self, PATH_FUNC, feature_)
                    with torch.no_grad():
                        reward = torch.tanh(getattr(self.mdp, PATH_FUNC).model(torch.cat([feature, feature_]))).item()
                    return np.append(super().get_reward(), reward)

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                encoding = self.describe()[coding]
                model = encoding['model']
                cls = lamarckian.evaluator.parse(*model['cls'], **kwargs)
                make = functools.partial(lambda cls, *args, **kwargs: cls(*args, **kwargs), cls, *args, **kwargs, **model)
                model = make()
                attr = types.SimpleNamespace(make=make, model=model, keys=model.state_dict().keys())
                attr.model.eval()
                setattr(self, PATH_FUNC, attr)

            def describe(self):
                encoding = super().describe()
                assert coding not in encoding
                encoding[coding] = self.describe_reward()
                encoding['blob']['reward'].append('blob')
                return encoding

            def initialize(self):
                decision = super().initialize()
                assert coding not in decision
                decision[coding] = lamarckian.model.to_blob(getattr(self, PATH_FUNC).make().state_dict())
                return decision

            def set(self, decision):
                attr = getattr(self, PATH_FUNC)
                attr.model.load_state_dict(lamarckian.model.from_blob(decision.pop(coding), attr.keys))
                return super().set(decision)

            def get(self):
                decision = super().get()
                assert coding not in decision
                decision[coding] = lamarckian.model.to_blob(getattr(self, PATH_FUNC).model.state_dict())
                return decision
        return MDP
    return decorate


def potential(get_feature, coding='reward_blob'):
    NAME_FUNC = inspect.getframeinfo(inspect.currentframe()).function
    PATH_FUNC = f'{__name__}.{NAME_FUNC}'

    def decorate(mdp):
        class MDP(mdp):
            class Controller(mdp.Controller):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    assert not hasattr(self, PATH_FUNC)

                def get_state(self):
                    state = super().get_state()
                    if not hasattr(self, PATH_FUNC):
                        setattr(self, PATH_FUNC, get_feature(self))
                    return state

                def get_reward(self):
                    feature = getattr(self, PATH_FUNC)
                    feature_ = get_feature(self)
                    setattr(self, PATH_FUNC, feature_)
                    model = getattr(self.mdp, PATH_FUNC).model
                    with torch.no_grad():
                        value = model(feature).item()
                        value_ = model(feature_).item()
                        reward = getattr(self.mdp, PATH_FUNC).discount * value_ - value
                    return np.append(super().get_reward(), reward)

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                assert not hasattr(self, PATH_FUNC)
                encoding = self.describe()[coding]
                model = encoding['model']
                cls = lamarckian.evaluator.parse(*model['cls'], **kwargs)
                make = functools.partial(lambda cls, *args, **kwargs: cls(*args, **kwargs), cls, *args, **kwargs, **model)
                model = make()
                attr = types.SimpleNamespace(make=make, model=model, keys=model.state_dict().keys(), discount=glom.glom(kwargs['config'], 'rl.discount'))
                attr.model.eval()
                setattr(self, PATH_FUNC, attr)

            def describe(self):
                encoding = super().describe()
                assert coding not in encoding
                encoding[coding] = self.describe_reward()
                encoding['blob']['reward'].append('blob')
                return encoding

            def initialize(self):
                decision = super().initialize()
                assert coding not in decision
                decision[coding] = lamarckian.model.to_blob(getattr(self, PATH_FUNC).make().state_dict())
                return decision

            def set(self, decision):
                attr = getattr(self, PATH_FUNC)
                attr.model.load_state_dict(lamarckian.model.from_blob(decision.pop(coding), attr.keys))
                return super().set(decision)

            def get(self):
                decision = super().get()
                assert coding not in decision
                decision[coding] = lamarckian.model.to_blob(getattr(self, PATH_FUNC).model.state_dict())
                return decision
        return MDP
    return decorate
