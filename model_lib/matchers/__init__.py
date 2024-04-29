# Copyright 2023 The Scenic Authors.
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
"""Matching utilities for Object Detection models."""

from model_lib.matchers.common import cpu_matcher, slicer
from model_lib.matchers.greedy import greedy_matcher
from model_lib.matchers.hungarian import hungarian_matcher
from model_lib.matchers.hungarian_cover import hungarian_cover_tpu_matcher
from model_lib.matchers.hungarian_jax import (hungarian_scan_tpu_matcher,
                                              hungarian_tpu_matcher)
from model_lib.matchers.lazy import lazy_matcher
from model_lib.matchers.sinkhorn import sinkhorn_matcher
