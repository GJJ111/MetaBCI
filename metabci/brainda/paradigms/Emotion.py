# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2020/6/01
# License: MIT License
"""
Emotion Paradigm.
"""
from .base import BaseParadigm


class Emotion(BaseParadigm):
    def is_valid(self, dataset):
        ret = True
        if dataset.paradigm != "Emotion":
            ret = False
        return ret
